import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .engine import make_template, run
from .judge import ModelConfig, ModelJudge

ROOT = Path(__file__).resolve().parent.parent


class App:
    def __init__(self, root):
        self.root = root
        root.title("车机音乐 Agent 评估工具")
        root.geometry("850x620")
        self.messages = queue.Queue()
        self.cancel_event = threading.Event()
        self.running = False
        self.input = tk.StringVar()
        self.output = tk.StringVar(value=str(ROOT / "outputs"))
        self.offline = tk.BooleanVar(value=True)
        self.url = tk.StringVar()
        self.model = tk.StringVar()
        self.key = tk.StringVar()
        self.key_env = "MUSIC_EVAL_API_KEY"
        self.timeout = 120
        self.temperature = 0
        self.status = tk.StringVar(value="请选择 Excel。仅规则检查模式无需配置模型。")
        self.latest_report = None
        box = ttk.Frame(root, padding=20)
        box.pack(fill="both", expand=True)
        ttk.Label(box, text="导入测试 Excel，输出评分表和整体报告", font=("Microsoft YaHei", 15)).pack(anchor="w", pady=8)
        ttk.Label(box, text="自动评估需要人工抽检。仅规则检查模式不会完成回复准确性及音乐质量评分。").pack(anchor="w")
        for title, variable, callback in [("测试Excel", self.input, self.choose_input), ("输出目录", self.output, self.choose_output)]:
            line = ttk.Frame(box)
            line.pack(fill="x", pady=7)
            ttk.Label(line, text=title, width=12).pack(side="left")
            ttk.Entry(line, textvariable=variable).pack(side="left", fill="x", expand=True)
            ttk.Button(line, text="选择", command=callback).pack(side="left", padx=5)
        ttk.Checkbutton(box, text="仅规则检查（不向模型发送数据，主观评分留空）", variable=self.offline).pack(anchor="w", pady=8)
        for title, variable, secret in [("API地址", self.url, False), ("模型名称", self.model, False), ("临时密钥", self.key, True)]:
            line = ttk.Frame(box)
            line.pack(fill="x", pady=5)
            ttk.Label(line, text=title, width=12).pack(side="left")
            ttk.Entry(line, textvariable=variable, show="*" if secret else "").pack(side="left", fill="x", expand=True)
        ttk.Label(box, text="API地址包含公司要求的路径，例如 https://网关地址/v1；临时密钥不写入文件。\n也可使用 MUSIC_EVAL_API_KEY 环境变量；网关无需密钥时可留空。").pack(anchor="w", pady=5)
        settings = ttk.Frame(box)
        settings.pack(fill="x", pady=5)
        ttk.Button(settings, text="读取配置", command=self.load_config).pack(side="left")
        ttk.Button(settings, text="保存配置（不含密钥）", command=self.save_config).pack(side="left", padx=5)
        ttk.Button(settings, text="生成Excel模板", command=self.template).pack(side="left", padx=5)
        controls = ttk.Frame(box)
        controls.pack(fill="x", pady=12)
        self.start_button = ttk.Button(controls, text="开始评估", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(controls, text="停止（完成当前请求后）", command=self.cancel_event.set).pack(side="left", padx=5)
        ttk.Button(controls, text="打开最新报告", command=self.open_report).pack(side="left", padx=5)
        self.bar = ttk.Progressbar(box, maximum=100)
        self.bar.pack(fill="x", pady=5)
        ttk.Label(box, textvariable=self.status, wraplength=790).pack(anchor="w", pady=10)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)
        config = ROOT / "config.local.json"
        if config.exists():
            self.apply_config(config)

    def choose_input(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            self.input.set(path)

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output.set(path)

    def apply_config(self, path):
        try:
            config = ModelConfig.load(path)
            config.endpoint()
            self.url.set(config.base_url)
            self.model.set(config.model)
            self.key_env = config.api_key_env
            self.timeout = config.timeout
            self.temperature = config.temperature
            self.status.set("已读取模型配置；取消勾选“仅规则检查”可进行模型辅助评估。")
        except (ValueError, OSError, TypeError):
            messagebox.showerror("配置错误", "无法读取配置；请参考config.example.json检查字段和地址。")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("模型配置", "*.json")])
        if path:
            self.apply_config(path)

    def current_config(self):
        return ModelConfig(self.url.get(), self.model.get(), self.key_env, self.timeout, self.temperature)

    def save_config(self):
        try:
            config = self.current_config()
            config.endpoint()
            path = filedialog.asksaveasfilename(initialfile="config.local.json", defaultextension=".json", filetypes=[("模型配置", "*.json")])
            if path:
                Path(path).write_text(json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
                self.status.set("配置已保存，未保存密钥。")
        except (ValueError, OSError):
            messagebox.showerror("配置错误", "请检查API地址、模型名称和文件写入权限。")

    def template(self):
        path = filedialog.asksaveasfilename(initialfile="评估输入模板.xlsx", defaultextension=".xlsx")
        if path:
            try:
                make_template(path)
                self.status.set("已生成模板，请替换示例行并填写真实关键帧。")
            except OSError:
                messagebox.showerror("写入失败", "请关闭Excel中打开的同名文件或选择其他目录。")

    def start(self):
        if self.running:
            return
        try:
            source = Path(self.input.get())
            if not source.is_file():
                raise ValueError("请先选择存在的Excel文件。")
            if not self.output.get().strip():
                raise ValueError("请选择输出目录。")
            judge = None if self.offline.get() else ModelJudge(self.current_config(), self.key.get())
        except ValueError as e:
            messagebox.showerror("尚未开始", str(e))
            return
        output = self.output.get()
        self.running = True
        self.cancel_event.clear()
        self.start_button.configure(state="disabled")
        self.status.set("评估中……每条模型请求可能需要数十秒。")
        self.bar["value"] = 0
        def worker():
            try:
                paths = run(source, output, judge,
                            progress=lambda done, total: self.messages.put(("progress", (done, total))),
                            cancelled=self.cancel_event.is_set)
                self.messages.put(("done", paths))
            except ValueError as e:
                self.messages.put(("error", str(e)))
            except Exception:
                # Avoid exposing HTTP bodies, secrets, or company records through exception text.
                self.messages.put(("error", "运行失败：请检查Excel格式、文件是否被占用及输出目录权限。"))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        while not self.messages.empty():
            kind, value = self.messages.get_nowait()
            if kind == "progress":
                done, total = value
                self.bar["value"] = done / total * 100
                self.status.set(f"已处理 {done}/{total} 条。")
            else:
                self.running = False
                self.start_button.configure(state="normal")
                if kind == "done":
                    self.latest_report = value[1]
                    state = "已停止，已处理记录已导出。" if self.cancel_event.is_set() else "处理结束，请检查报告中的待定和待复核项。"
                    self.status.set(f"{state}\n评分表：{value[0]}\n报告：{value[1]}")
                else:
                    self.status.set(value)
                    messagebox.showerror("无法完成评估", value)
        self.root.after(100, self.poll)

    def open_report(self):
        if self.latest_report:
            import webbrowser
            webbrowser.open(self.latest_report.resolve().as_uri())

    def close(self):
        if self.running:
            messagebox.showinfo("正在评估", "请先点停止，等待当前请求结束并保存结果后再关闭。")
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
