"""Optional command-line interface for developers and verification."""
import argparse
from music_eval.engine import make_template, run
from music_eval.judge import ModelConfig, ModelJudge


def main():
    parser = argparse.ArgumentParser(description="车机音乐Agent Excel评估")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--config", help="模型配置；不传时只做规则检查")
    parser.add_argument("--template", help="生成模板到指定路径")
    args = parser.parse_args()
    if args.template:
        make_template(args.template)
        print("模板已生成：", args.template)
        return
    if not args.input:
        parser.error("请提供输入Excel或--template")
    judge = ModelJudge(ModelConfig.load(args.config)) if args.config else None
    for path in run(args.input, args.output, judge, progress=lambda d,t: print(f"已处理 {d}/{t}")):
        print(path)


if __name__ == "__main__":
    main()
