"""Command-line entry point: ``safedpo {build-data,train,eval,report}``."""

from __future__ import annotations

import argparse
import json
import logging

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="safedpo", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-data", help="construct safe/unsafe preference pairs")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml", help="used for decontamination")
    p.add_argument("--stage", default="all", choices=["all", "collect", "generate", "judge", "pair"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")

    p = sub.add_parser("train", help="LoRA + DPO training")
    p.add_argument("--config", default="configs/dpo_llama3_8b.yaml")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")

    p = sub.add_parser("eval", help="evaluate a model / adapter")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--tag", required=True, help="results/<tag>/")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="path to a LoRA adapter (omit for the base model)")
    p.add_argument("--stages", default="generate,judge,mmlu,metrics")
    p.add_argument("--judge-benches", default=None, help="e.g. harmbench or jbb (default: all)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")

    p = sub.add_parser("report", help="aggregate metrics into tables / figure / README")
    p.add_argument("--results", default="results")
    p.add_argument("--tags", nargs="+", default=["base", "dpo"], help="first tag is the baseline")
    p.add_argument("--update-readme", default=None, metavar="README.md")
    p.add_argument("--data-stats", default=None, metavar="data/processed/stats.json")
    p.add_argument("--train-log", default=None, metavar="outputs/.../log_history.json")
    p.add_argument("--marker", default="RESULTS", help="README section marker name")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for noisy in ("httpx", "httpx2", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "build-data":
        from .data.build import run

        cfg = load_config(args.config, args.override)
        eval_cfg = load_config(args.eval_config) if cfg.get("decontaminate", {}).get("enabled") else None
        stats = run(cfg, eval_cfg, stage=args.stage, overwrite=args.overwrite)
        if stats:
            print(json.dumps(stats, indent=2, ensure_ascii=False))
    elif args.cmd == "train":
        from .train import train

        print(train(load_config(args.config, args.override)))
    elif args.cmd == "eval":
        from .eval.run import run_eval

        m = run_eval(
            load_config(args.config, args.override),
            args.tag,
            args.model,
            args.adapter,
            stages=[s.strip() for s in args.stages.split(",")],
            overwrite=args.overwrite,
            judge_benches=args.judge_benches.split(",") if args.judge_benches else None,
        )
        if m:
            print(json.dumps({k: v for k, v in m.items() if k != "model"}, indent=2)[:4000])
    elif args.cmd == "report":
        from .report import build_report

        print(
            build_report(
                args.results, args.tags, args.update_readme, args.data_stats, args.train_log, args.marker
            )
        )


if __name__ == "__main__":
    main()
