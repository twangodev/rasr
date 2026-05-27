"""Real WER eval of an ASR checkpoint on ATCO2 validation (CPU only).

The training-loop validation WER reported in checkpoint filenames (e.g.
"step026280-wer0.1511") is digit-NAIVE: it scores "two one zero" against "210"
as three substitutions, inflating the number. This script recomputes WER with
the project's proper metrics on the SAME greedy hypotheses:

  * canonical WER     (lowercase + de-punct; the metric BENCHMARKS.md uses)
  * digit-aware WER    ("two one zero" <-> "210"; the honest metric)
  * numeric-only WER   (WER over only the numeric tokens; ATC-safety content)

Model loading supports two inputs via --model:
  * a *.nemo  -> ASRModel.restore_from
  * a hub id  -> ASRModel.from_pretrained
  * a Lightning *.ckpt (full state) -> load the base pretrained model named by
    --base (default nvidia/parakeet-tdt-0.6b-v3) for its tokenizer/config, then
    load_state_dict the ckpt's ["state_dict"]. (Direct load_from_checkpoint
    fails on these ckpts: the hparams cfg points at a tokenizer artifact path
    that only existed inside the original .nemo tar.)

CPU ONLY. A training job owns the GPU. Always run with CUDA_VISIBLE_DEVICES="".

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/eval_real_wer.py \
        --model ckpt/parakeet-higgs-mixed/step026280-wer0.1511.ckpt
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
    """Load a model onto CPU.

    Handles .nemo, hub ids, and Lightning .ckpt (full-state) checkpoints. For a
    .ckpt we instantiate the `base` pretrained model (tokenizer + config) and
    load the checkpoint's state_dict into it.
    """
    if path.endswith(".ckpt") and os.path.exists(path):
        print(f"[load] .ckpt detected; loading base '{base}' for tokenizer/config ...", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(base, map_location="cpu")
        model = model.cpu()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ck["state_dict"]
        # The Lightning .ckpt stores NeMo native keys (encoder./decoder./joint./
        # preprocessor.) with no extra wrapper prefix, so no stripping is needed.
        # Stay defensive in case a future ckpt wraps under "model.".
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
    parser.add_argument("--limit", type=int, default=0, help="cap clips (0 = all 113)")
    parser.add_argument("--batch-size", type=int, default=8, help="transcribe batch size (CPU)")
    args = parser.parse_args()

    from datasets import load_dataset

    model = _load_model(args.model, args.base)
    print(
        f"[load] type={type(model).__name__} "
        f"default_strategy={model.cfg.decoding.get('strategy')}",
        flush=True,
    )

    print("[data] loading jlvdoorn/atco2-asr:validation ...", flush=True)
    ds = load_dataset("jlvdoorn/atco2-asr", split="validation")
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    picked = list(range(n))
    audios = [ds[i]["audio"]["array"].astype("float32") for i in picked]
    refs = [ds[i]["text"].strip() for i in picked]

    # Sanity transcription of one clip before the full run.
    print("\n[sanity] transcribing clip 0 ...", flush=True)
    sanity = _text(model.transcribe(audio=[audios[0]], batch_size=1, verbose=False)[0])
    print(f"  REF : {refs[0]}", flush=True)
    print(f"  HYP : {sanity}", flush=True)

    print(f"\n[decode] greedy, no biasing, {n} clips, batch_size={args.batch_size} ...", flush=True)
    hyps = [_text(h) for h in model.transcribe(audio=audios, batch_size=args.batch_size, verbose=False)]

    canonical = corpus_wer(refs, hyps)
    digit_aware = wer_digit_aware(refs, hyps)
    numeric = corpus_numeric_wer(refs, hyps)

    print("\n" + "=" * 72)
    print(f"REAL WER  --  {args.model}")
    print("=" * 72)
    print(f"  N clips                 : {n}")
    print(f"  canonical WER           : {canonical:.4f}")
    print(f"  digit-aware WER         : {digit_aware:.4f}")
    print(f"  numeric-only WER        : {numeric:.4f}")
    print("-" * 72)
    print("  reference points:")
    print("    training-val (digit-naive)        : 0.1511")
    print("    rasr-v1 canonical (BENCHMARKS)     : 0.125")
    print("    parakeet-mixed-bandpass canonical  : 0.130")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
