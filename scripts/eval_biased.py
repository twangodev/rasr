"""Biased-vs-unbiased contextual-biasing eval for a finetuned TDT ATC model.

For each of the first N ATCO2 validation clips, transcribe twice:

  * UNBIASED: the model's default greedy_batch decoding, no boosting tree.
  * BIASED:   greedy_batch with a per-clip boosting tree built from
              `bias_terms_from_info(info)` (waypoints + airline telephony
              names), weighted by `--alpha`.

Both hypotheses are scored with the digit-aware WER (so "two one zero" vs
"210" doesn't pollute the comparison) and, for reference, the existing
canonical WER. Per-clip and corpus aggregates are printed, plus example diffs.

CPU ONLY. A training job owns the GPU. Run with `CUDA_VISIBLE_DEVICES=""`.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/eval_biased.py \
        --model ckpt/parakeet-mixed-bandpass/final.nemo --limit 6 --alpha 2.0
"""

from __future__ import annotations

import argparse
import os
import sys

# Belt-and-suspenders: force CPU even if the caller forgot the env var.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nemo.collections.asr as nemo_asr  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from rasr.eval.context import bias_terms_from_info  # noqa: E402
from rasr.eval.metrics.wer import (  # noqa: E402
    utt_wer,
    utt_wer_digit_aware,
    wer_digit_aware,
)
from rasr.eval.metrics import corpus_wer  # noqa: E402

# NeMo's boosting-tree example uses depth_scaling 2.0 for CTC/RNN-T/TDT and a
# unit context_score; alpha is the fusion weight we sweep.
_BIAS_LIST_CAP = 64  # cap terms/clip unless --no-bias-list-cap (keeps tree small)


def _load_model(path: str):
    """Load a .nemo finetune (or hub id) onto CPU as an ASRModel."""
    if path.endswith(".nemo") and os.path.exists(path):
        model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(path, map_location="cpu")
    model = model.cpu()
    model.eval()
    return model


def _unbiased_cfg(base_decoding_cfg):
    """greedy_batch decoding with NO boosting tree."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_decoding_cfg, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        if "greedy" not in cfg or cfg.greedy is None:
            cfg.greedy = {}
        cfg.greedy.boosting_tree = None
        cfg.greedy.boosting_tree_alpha = None
    return cfg


def _biased_cfg(base_decoding_cfg, bias_terms: list[str], alpha: float):
    """greedy_batch decoding with a boosting tree built from `bias_terms`."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_decoding_cfg, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        if "greedy" not in cfg or cfg.greedy is None:
            cfg.greedy = {}
        cfg.greedy.boosting_tree = {
            "key_phrases_list": list(bias_terms),
            "context_score": 1.0,
            "depth_scaling": 2.0,
            "use_triton": False,  # CPU: never Triton.
        }
        cfg.greedy.boosting_tree_alpha = alpha
    return cfg


def _text(hyp) -> str:
    return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help=".nemo path or hub id")
    parser.add_argument("--limit", type=int, default=6, help="number of clips")
    parser.add_argument("--alpha", type=float, default=2.0, help="boosting alpha")
    parser.add_argument(
        "--no-bias-list-cap",
        action="store_true",
        help="do not cap the per-clip bias list at %d terms" % _BIAS_LIST_CAP,
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[load] {args.model} on CPU ...", flush=True)
    model = _load_model(args.model)
    print(
        f"[load] type={type(model).__name__} "
        f"default_strategy={model.cfg.decoding.get('strategy')}",
        flush=True,
    )
    base_decoding_cfg = model.cfg.decoding

    print("[data] loading jlvdoorn/atco2-asr:validation ...", flush=True)
    ds = load_dataset("jlvdoorn/atco2-asr", split="validation")
    n = min(args.limit, len(ds))
    picked = list(range(n))

    audios = [ds[i]["audio"]["array"].astype("float32") for i in picked]
    refs = [ds[i]["text"].strip() for i in picked]
    bias_lists = []
    for i in picked:
        terms = bias_terms_from_info(ds[i]["info"])
        if not args.no_bias_list_cap:
            terms = terms[:_BIAS_LIST_CAP]
        bias_lists.append(terms)

    # --- Unbiased baseline (single batch, default decoding) ---
    print("\n[decode] unbiased (no boosting) ...", flush=True)
    model.change_decoding_strategy(_unbiased_cfg(base_decoding_cfg))
    unbiased_hyps = [_text(h) for h in model.transcribe(audio=audios, batch_size=1, verbose=False)]

    # --- Biased: per-clip boosting tree (per-clip context is the real setting) ---
    print("[decode] biased (per-clip boosting tree) ...", flush=True)
    biased_hyps = []
    for idx, (a, terms) in enumerate(zip(audios, bias_lists)):
        if terms:
            model.change_decoding_strategy(_biased_cfg(base_decoding_cfg, terms, args.alpha))
        else:
            model.change_decoding_strategy(_unbiased_cfg(base_decoding_cfg))
        out = model.transcribe(audio=[a], batch_size=1, verbose=False)
        biased_hyps.append(_text(out[0]))
        print(f"  clip {picked[idx]}: {len(terms)} bias terms", flush=True)
    model.change_decoding_strategy(base_decoding_cfg)  # restore clean state

    # --- Report ---
    print("\n" + "=" * 88)
    print(f"PER-CLIP RESULTS  (alpha={args.alpha}, digit-aware WER; canonical in parens)")
    print("=" * 88)
    for n_, i in enumerate(picked):
        ub = utt_wer_digit_aware(refs[n_], unbiased_hyps[n_])
        bi = utt_wer_digit_aware(refs[n_], biased_hyps[n_])
        ub_c = utt_wer(refs[n_], unbiased_hyps[n_])
        bi_c = utt_wer(refs[n_], biased_hyps[n_])
        flag = ""
        if bi < ub:
            flag = "  <-- biasing HELPS"
        elif bi > ub:
            flag = "  <-- biasing HURTS"
        print(
            f"\n--- clip {i}  WER unbiased={ub:.3f}  biased={bi:.3f}  "
            f"delta={bi - ub:+.3f}  (canon {ub_c:.3f}/{bi_c:.3f}){flag}"
        )
        print(f"  REF     : {refs[n_]}")
        print(f"  UNBIASED: {unbiased_hyps[n_]}")
        print(f"  BIASED  : {biased_hyps[n_]}")
        print(f"  bias[:12]: {bias_lists[n_][:12]}")

    print("\n" + "=" * 88)
    print("AGGREGATE")
    print("=" * 88)
    ub_corpus = wer_digit_aware(refs, unbiased_hyps)
    bi_corpus = wer_digit_aware(refs, biased_hyps)
    print(f"  unbiased corpus WER (digit-aware): {ub_corpus:.4f}")
    print(f"  biased   corpus WER (digit-aware): {bi_corpus:.4f}")
    print(f"  delta (biased - unbiased)        : {bi_corpus - ub_corpus:+.4f}")
    print(f"  unbiased corpus WER (canonical)  : {corpus_wer(refs, unbiased_hyps):.4f}")
    print(f"  biased   corpus WER (canonical)  : {corpus_wer(refs, biased_hyps):.4f}")
    print(f"  clips={n}  alpha={args.alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
