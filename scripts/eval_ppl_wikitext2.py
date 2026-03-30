
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


def _log(msg: str) -> None:
    print(msg, flush=True)


MINDCHAIN_WIKITEXT2 = "mindchain/wikitext2"
SPARSEGPT_RAW_WIKITEXT2_CONFIG = "wikitext-2-raw-v1"
SPARSEGPT_WIKITEXT_SPLIT = "test"




def _load_wikitext_test_split(dataset_id: str, config: str, token: str | None):
    if dataset_id == MINDCHAIN_WIKITEXT2:
        _log(
            f"Using dataset content for {MINDCHAIN_WIKITEXT2!r} via canonical "
            f"`wikitext` + {config!r} (same raw WikiText-2 files; `datasets` 4.x cannot load hub scripts)."
        )
        return load_dataset("wikitext", config, split=SPARSEGPT_WIKITEXT_SPLIT)
    return load_dataset(dataset_id, config, split=SPARSEGPT_WIKITEXT_SPLIT, token=token)


def _local_dir_has_tokenizer(model_path: str) -> bool:
    p = Path(model_path)
    if not p.is_dir():
        return True  
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
        default=0,
        help=(
            "Non-overlapping chunk limit (each chunk is seq_len tokens). "
            "Default 0 = use the **full** joined test set. Set a positive integer to cap cost."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional JSON path under results/ (default: results/ppl_<safe_model>.json)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="mindchain/wikitext2",
        help="Hugging Face dataset id (default: mindchain/wikitext2)",
    )
    p.add_argument(
        "--dataset-config",
        type=str,
        default=SPARSEGPT_RAW_WIKITEXT2_CONFIG,
        help=(
            "Must be wikitext-2-raw-v1 for raw-WikiText-2 (SparseGPT paper / HF). "
            "Do not use wikitext-2-v1 here — that is the tokenized '<unk>' variant."
        ),
    )
    args = p.parse_args()

    if args.dataset_config != SPARSEGPT_RAW_WIKITEXT2_CONFIG:
        _log(
            f"WARNING: dataset-config is {args.dataset_config!r}, not {SPARSEGPT_RAW_WIKITEXT2_CONFIG!r}. "
            "SparseGPT reports perplexity on **raw** WikiText-2 (test)."
        )

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
    _log(f"device={device} dtype={dtype}")

    _log(f"Loading tokenizer from {tok_src!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        token=token,
        use_fast=True,
        trust_remote_code=True,
    )
    _log("Loading model weights (can take several minutes) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=token,
    )
    model.eval()
    _log("Model ready.")

    _log(
        f"Loading raw-WikiText-2 test split ({args.dataset!r}, config={args.dataset_config!r}, "
        f"split={SPARSEGPT_WIKITEXT_SPLIT!r}) ..."
    )
    ds = _load_wikitext_test_split(args.dataset, args.dataset_config, token)
    # HF / SparseGPT (Appendix B): join test articles with newlines for the standard PPL recipe.
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    nlls: list[float] = []
    total_chunks = max(0, (input_ids.numel() - 1) // args.seq_len)
    if args.max_chunks <= 0:
        num_chunks = total_chunks
    else:
        num_chunks = min(args.max_chunks, total_chunks)
    if args.max_chunks > 0 and num_chunks < total_chunks:
        _log(
            f"NOTE: Capped at {num_chunks}/{total_chunks} chunks (--max-chunks={args.max_chunks}). "
            "Omit --max-chunks or use --max-chunks 0 for the full test set."
        )
    _log(
        f"seq_len={args.seq_len} max_chunks={'full' if args.max_chunks <= 0 else args.max_chunks} "
        f"-> running {num_chunks} forward passes ..."
    )

    progress_every = max(1, min(32, num_chunks // 8 or 1))
    with torch.no_grad():
        for i in range(num_chunks):
            if i == 0 or (i + 1) % progress_every == 0 or i + 1 == num_chunks:
                _log(f"  chunk {i + 1}/{num_chunks}")
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
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_loader_used": (
            f"wikitext/{args.dataset_config}"
            if args.dataset == MINDCHAIN_WIKITEXT2
            else f"{args.dataset}/{args.dataset_config}"
        ),
        "split": SPARSEGPT_WIKITEXT_SPLIT,
        "protocol": (
            "raw-WikiText-2 test (HF wikitext-2-raw-v1); concatenate examples with "
            "double-newline separators; non-overlapping seq_len windows (SparseGPT Appendix B; HF perplexity doc)."
        ),
        "seq_len": args.seq_len,
        "max_chunks": args.max_chunks,
        "num_chunks": len(nlls),
        "num_chunks_total_available": total_chunks,
        "full_split_evaluated": len(nlls) == total_chunks,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "device": device,
        "dtype": str(dtype),
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    _log(f"Wrote {out_path}")
    _log(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
