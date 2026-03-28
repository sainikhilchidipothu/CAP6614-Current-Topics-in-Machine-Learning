
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_token import load_hf_token


def _local_dir_has_tokenizer(model_path: str) -> bool:
    p = Path(model_path)
    if not p.is_dir():
        return True  # HF hub id — tokenizer comes from the repo
    return any(
        (p / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="HF id or path for tokenizer when --model is a weight-only checkpoint "
        "(e.g. meta-llama/Llama-2-7b-hf). Defaults to --model if that folder has tokenizer files.",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=256,
        help="Max non-overlapping chunks of seq_len tokens (caps eval cost)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional JSON path under results/ (default: results/ppl_<safe_model>.json)",
    )
    args = p.parse_args()

    tok_src = args.tokenizer.strip() or args.model
    if not args.tokenizer.strip() and not _local_dir_has_tokenizer(args.model):
        sys.exit(
            "This model path has no tokenizer files (only weights were saved). "
            "Pass the original HF model id, e.g.\n"
            "  --tokenizer meta-llama/Llama-2-7b-hf"
        )

    token = load_hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        token=token,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=token,
    )
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    nlls: list[float] = []
    num_chunks = min(args.max_chunks, max(0, (input_ids.numel() - 1) // args.seq_len))

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * args.seq_len
            end = start + args.seq_len
            chunk = input_ids[start:end].unsqueeze(0)
            labels = chunk.clone()
            if device == "cuda":
                chunk = chunk.to("cuda")
                labels = labels.to("cuda")
            out = model(chunk, labels=labels)
            nlls.append(float(out.loss))

    mean_nll = sum(nlls) / len(nlls) if nlls else float("nan")
    ppl = math.exp(mean_nll) if nlls else float("nan")

    safe = args.model.replace("/", "__")
    out_path = Path(args.out) if args.out else Path("results") / f"ppl_{safe}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": args.model,
        "tokenizer": tok_src,
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "seq_len": args.seq_len,
        "num_chunks": len(nlls),
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "device": device,
        "dtype": str(dtype),
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
