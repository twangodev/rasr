"""Generalized WER eval of an ASR model on an arbitrary HF ASR dataset (CPU only).

Computes the project's canonical and digit-aware WER on greedy hypotheses for a
given model over a configurable dataset / split / clip cap. Built to evaluate
the radiotalk-asr ATC models on a second test set (e.g. jacktol/ATC-ASR-Dataset)
alongside the European ATCO2 benchmark.

Model loading (mirrors scripts/eval_real_wer.py):
  * a *.nemo   -> ASRModel.restore_from
  * a hub id   -> ASRModel.from_pretrained
  * a Lightning *.ckpt (full state) -> load the base pretrained model named by
    --base (default nvidia/parakeet-tdt-0.6b-v3) for its tokenizer/config, then
    load_state_dict the ckpt's ["state_dict"].

CPU ONLY. A training job owns the GPU. Always run with CUDA_VISIBLE_DEVICES="".

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/eval_us.py \
        --model ckpt/parakeet-mixed/final.nemo \
        --dataset jacktol/ATC-ASR-Dataset --split test --limit 300
"""

from __future__ import annotations

import argparse
import os
import sys

# Belt-and-suspenders: force CPU even if the caller forgot the env var.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import nemo.collections.asr as nemo_asr  # noqa: E402

from rasr.eval.metrics.wer import (  # noqa: E402
    corpus_numeric_wer,
    corpus_wer,
    wer_digit_aware,
)


def _load_model(path: str, base: str):
    """Load a model onto CPU (.nemo, hub id, or Lightning full-state .ckpt)."""
    if path.endswith(".ckpt") and os.path.exists(path):
        print(f"[load] .ckpt detected; loading base '{base}' for tokenizer/config ...", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(base, map_location="cpu")
        model = model.cpu()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ck["state_dict"]
        if state_dict and all(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
        res = model.load_state_dict(state_dict, strict=False)
        print(
            f"[load] load_state_dict: missing={len(res.missing_keys)} "
            f"unexpected={len(res.unexpected_keys)}",
            flush=True,
        )
        if res.missing_keys:
            print(f"[load]   missing[:8]={res.missing_keys[:8]}", flush=True)
        if res.unexpected_keys:
            print(f"[load]   unexpected[:8]={res.unexpected_keys[:8]}", flush=True)
    elif path.endswith(".nemo") and os.path.exists(path):
        model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu")
        model = model.cpu()
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(path, map_location="cpu")
        model = model.cpu()
    model.eval()
    return model


def _text(hyp) -> str:
    return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help=".ckpt / .nemo path or hub id")
    parser.add_argument(
        "--base",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="base pretrained model used to load a .ckpt's state_dict",
    )
    parser.add_argument("--dataset", required=True, help="HF dataset id")
    parser.add_argument("--split", default="test", help="dataset split")
    parser.add_argument("--limit", type=int, default=0, help="cap clips (0 = all)")
    parser.add_argument("--batch-size", type=int, default=8, help="transcribe batch size (CPU)")
    args = parser.parse_args()

    from datasets import load_dataset

    model = _load_model(args.model, args.base)
    print(
        f"[load] type={type(model).__name__} "
        f"default_strategy={model.cfg.decoding.get('strategy')}",
        flush=True,
    )

    print(f"[data] loading {args.dataset}:{args.split} ...", flush=True)
    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    picked = list(range(n))
    audios = [ds[i]["audio"]["array"].astype("float32") for i in picked]
    refs = [ds[i]["text"].strip() for i in picked]

    print("\n[sanity] transcribing clip 0 ...", flush=True)
    sanity = _text(model.transcribe(audio=[audios[0]], batch_size=1, verbose=False)[0])
    print(f"  REF : {refs[0]}", flush=True)
    print(f"  HYP : {sanity}", flush=True)

    print(f"\n[decode] greedy, {n} clips, batch_size={args.batch_size} ...", flush=True)
    hyps = [_text(h) for h in model.transcribe(audio=audios, batch_size=args.batch_size, verbose=False)]

    canonical = corpus_wer(refs, hyps)
    digit_aware = wer_digit_aware(refs, hyps)
    numeric = corpus_numeric_wer(refs, hyps)

    print("\n" + "=" * 72)
    print(f"WER  --  model={args.model}")
    print(f"        dataset={args.dataset}:{args.split}")
    print("=" * 72)
    print(f"  N clips                 : {n}")
    print(f"  canonical WER           : {canonical:.4f}")
    print(f"  digit-aware WER         : {digit_aware:.4f}")
    print(f"  numeric-only WER        : {numeric:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
