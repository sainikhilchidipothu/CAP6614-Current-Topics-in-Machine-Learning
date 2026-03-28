from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hf_token import ensure_hf_token_in_env

from lm_eval import simple_evaluate


def _local_dir_has_tokenizer(model_path: str) -> bool:
    p = Path(model_path)
    if not p.is_dir():
        return True
    return any(
        (p / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )


def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Zero-shot benchmarks (default tasks match common LLM compression papers)."
    )
    p.add_argument("--model", type=str, required=True, help="HF model id or local path")
    p.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="HF id for tokenizer when --model is weights-only (e.g. meta-llama/Llama-2-7b-hf).",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="piqa,arc_easy,arc_challenge",
        help="Comma-separated lm-eval task names",
    )
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--batch-size", type=str, default="auto")
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Fraction or count of examples per task (for quick smoke tests)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="Bootstrap iters for stderr; 0 is faster",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: results/zeroshot_<safe_model>.json)",
    )
    args = p.parse_args()

    ensure_hf_token_in_env()

    tok = args.tokenizer.strip()
    if not tok and not _local_dir_has_tokenizer(args.model):
        sys.exit(
            "This model path has no tokenizer files. Pass e.g.\n"
            "  --tokenizer meta-llama/Llama-2-7b-hf"
        )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_args = f"pretrained={args.model},dtype=float16,trust_remote_code=True"
    if tok:
        model_args += f",tokenizer={tok}"

    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=False,
    )

    safe = args.model.replace("/", "__").replace(" ", "_")
    out_path = Path(args.out) if args.out else Path("results") / f"zeroshot_{safe}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(results)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload.get("results", payload), indent=2))


if __name__ == "__main__":
    main()
