"""Spike: decode-time contextual biasing for ATC ASR on ATCO2.

Goal: prove that NeMo's transducer context-biasing (GPU boosting tree, used as a
`fusion_model` in `greedy_batch` TDT decoding) can pull per-clip context terms
(waypoints, airline telephony names) into the output of
`nvidia/parakeet-tdt-0.6b-v3`, reducing WER on the rare/OOD word error class.

CPU ONLY. Run with `CUDA_VISIBLE_DEVICES=""`. This is a spike, not production
code: it touches only a handful of clips, prints diffs, and reports WER.

Mechanism (the pinned NeMo API for this model):
  - parakeet-tdt-0.6b-v3 is an EncDecRNNTBPEModel (TDT transducer) WITH an aux
    CTC head. The cleanest supported biasing path is the *transducer* boosting
    tree, not CTC-WS: NeMo wires a `GPUBoostingTreeModel` (built from a plain
    word list) into TDT `greedy_batch` decoding as a fusion model.
  - rnnt_decoding.py only accepts `boosting_tree` for the GREEDY_BATCH strategy
    (TDT/RNNT). `greedy`/`beam`/`maes` raise NotImplementedError. greedy_batch is
    already this model's default strategy.
  - On CPU the boosting tree's `advance()` uses the pure-PyTorch path
    (`use_triton and device=='cuda'` gates Triton), so no GPU is required.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/spikes/context_biasing_spike.py

Spike findings (pretrained nvidia/parakeet-tdt-0.6b-v3, CPU, 5 ATCO2 val clips,
boosting_tree_alpha=3.0):
  - parakeet-tdt-0.6b-v3 IS a TDT transducer (EncDecRNNTBPEModel) WITH an aux CTC
    head. The biasing path used here is the TDT *boosting tree* fusion model in
    `greedy_batch` decoding (NeMo only accepts boosting_tree for greedy_batch).
  - The mechanism works end-to-end on CPU: the GPUBoostingTreeModel is built from
    a plain word list and attached as a fusion model; output demonstrably shifts
    toward context terms (e.g. "Praha Ruzyne", "Sky Travel", "Czech").
  - BUT the net WER effect on the pretrained v3 is small and fragile. Corpus WER
    over the 5 clips went 0.728 -> 0.680 at alpha=3.0, but per-clip it both fixes
    real terms AND hallucinates waypoints that don't sound like the audio
    (e.g. "RISUK", "ERASU7", "ARVEGRO"). An alpha sweep (1.0/2.0/3.0) showed no
    clean monotone win. Much of the absolute WER (~0.7) is a *measurement
    artifact*: the repo's canonical_transform does not normalize "210" vs
    "two one zero", so digits inflate WER on both base and biased equally.
  - Honest read: biasing is a plausible win but needs (a) a digit-aware metric,
    (b) per-clip alpha tuning, (c) ADS-B->telephony callsign expansion, and ideally
    (d) the *finetuned* rasr model rather than the raw multilingual v3.
"""

from __future__ import annotations

import os
import sys

# Belt-and-suspenders: force CPU even if the caller forgot the env var.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import nemo.collections.asr as nemo_asr  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

# Repo WER (canonical_transform: lowercase, strip punct, collapse spaces).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rasr.eval.metrics.wer import corpus_wer, utt_wer  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
# How hard to boost. NeMo example default for transducer is ~3.0; ATC context
# lists are short and high-precision, so a moderate weight is a reasonable start.
BOOST_ALPHA = 3.0
NUM_CLIPS = 5


def parse_bias_terms(info: str) -> list[str]:
    """Parse an ATCO2 `info` field into decode-time bias phrases.

    `info` is newline-separated:
      0: airport ICAO (e.g. LKPR)        -> skip (not spoken as a word)
      1: airport name (e.g. Praha Ruzyne)-> include (sometimes spoken)
      2: sector (e.g. Radar)             -> include
      3: waypoint fixes (AKEVA ARVEG ...)-> include, spoken verbatim
      4: ADS-B callsign codes (EWG7AB ...)-> SKIP for verbatim biasing: these are
         NOT spoken as written (EWG7AB -> "Eurowings Seven Alfa Bravo"). Verbatim
         biasing them is useless; telephony expansion would be needed (see report).
      5: airline telephony names (Eurowings Wizzair ...) -> include, spoken form
    """
    lines = info.split("\n")
    terms: list[str] = []

    def add_words(line: str) -> None:
        for w in line.split():
            w = w.strip()
            # Single-token, alpha-ish bias words only (skip pure-numeric noise).
            if w and not w.isdigit():
                terms.append(w)

    if len(lines) > 1 and lines[1].strip():
        terms.append(lines[1].strip())  # airport name (multi-word phrase)
    if len(lines) > 2 and lines[2].strip():
        add_words(lines[2])  # sector
    if len(lines) > 3 and lines[3].strip():
        add_words(lines[3])  # waypoints (verbatim spoken)
    if len(lines) > 5 and lines[5].strip():
        # Airline names line: split into individual telephony words. Keep the
        # "Csa / Czech" style slashes out.
        for w in lines[5].replace("/", " ").split():
            if w.strip():
                terms.append(w.strip())

    # De-dup, preserve order, drop 1-char fragments.
    seen, out = set(), []
    for t in terms:
        k = t.lower()
        if k not in seen and len(t) >= 2:
            seen.add(k)
            out.append(t)
    return out


def build_boosting_decoding_cfg(base_decoding_cfg, bias_terms: list[str], alpha: float):
    """Return a decoding cfg (copy of model default) with a boosting tree attached."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_decoding_cfg, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        if "greedy" not in cfg or cfg.greedy is None:
            cfg.greedy = {}
        # BoostingTreeModelConfig fields consumed by rnnt_decoding.py.
        cfg.greedy.boosting_tree = {
            "key_phrases_list": list(bias_terms),
            "context_score": 1.0,
            "depth_scaling": 2.0,  # 2.0 for CTC/RNN-T/TDT per NeMo
            "use_triton": False,   # CPU: never use Triton
        }
        cfg.greedy.boosting_tree_alpha = alpha
    return cfg


def main() -> int:
    from datasets import load_dataset

    print(f"[load] {MODEL_ID} on CPU ...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID, map_location="cpu")
    model = model.cpu()
    model.eval()
    print(f"[load] type={type(model).__name__} aux_ctc={model.cfg.get('aux_ctc') is not None} "
          f"default_strategy={model.cfg.decoding.get('strategy')}", flush=True)

    base_decoding_cfg = model.cfg.decoding

    print("[data] loading jlvdoorn/atco2-asr:validation ...", flush=True)
    ds = load_dataset("jlvdoorn/atco2-asr", split="validation")

    # Pick clips that actually have a rich info block (waypoints + airlines),
    # so biasing has something to bite on. Skip ones with empty context.
    picked = []
    for i in range(len(ds)):
        terms = parse_bias_terms(ds[i]["info"])
        if len(terms) >= 8:
            picked.append(i)
        if len(picked) >= NUM_CLIPS:
            break
    print(f"[data] selected clip indices: {picked}", flush=True)

    audios = [ds[i]["audio"]["array"].astype("float32") for i in picked]
    refs = [ds[i]["text"].strip() for i in picked]
    bias_lists = [parse_bias_terms(ds[i]["info"]) for i in picked]

    # --- Baseline: default greedy_batch decoding, no biasing ---
    print("\n[decode] baseline (no biasing) ...", flush=True)
    base_hyps = model.transcribe(audio=audios, batch_size=1, verbose=False)
    base_hyps = [(h.text if hasattr(h, "text") else str(h)).strip() for h in base_hyps]

    # --- Biased: per-clip boosting tree. greedy_batch fusion is per-batch, so we
    # rebuild decoding per clip with that clip's own context list and decode it
    # alone (this is the realistic per-clip-context setting). ---
    print("[decode] biased (per-clip boosting tree) ...", flush=True)
    biased_hyps = []
    for idx, (a, terms) in enumerate(zip(audios, bias_lists)):
        cfg = build_boosting_decoding_cfg(base_decoding_cfg, terms, BOOST_ALPHA)
        model.change_decoding_strategy(cfg)
        out = model.transcribe(audio=[a], batch_size=1, verbose=False)
        biased_hyps.append((out[0].text if hasattr(out[0], "text") else str(out[0])).strip())
        print(f"  clip {picked[idx]}: {len(terms)} bias terms", flush=True)
    # restore default decoding (clean state)
    model.change_decoding_strategy(base_decoding_cfg)

    # --- Report ---
    print("\n" + "=" * 80)
    print("PER-CLIP RESULTS")
    print("=" * 80)
    base_wers, biased_wers = [], []
    for n, i in enumerate(picked):
        bw = utt_wer(refs[n], base_hyps[n])
        cw = utt_wer(refs[n], biased_hyps[n])
        base_wers.append(bw)
        biased_wers.append(cw)
        print(f"\n--- clip {i}  WER base={bw:.3f}  biased={cw:.3f}  delta={cw - bw:+.3f}")
        print(f"  REF   : {refs[n]}")
        print(f"  BASE  : {base_hyps[n]}")
        print(f"  BIASED: {biased_hyps[n]}")
        print(f"  bias[:12]: {bias_lists[n][:12]}")

    print("\n" + "=" * 80)
    print("AGGREGATE (corpus WER over the handful)")
    print("=" * 80)
    print(f"  baseline corpus WER : {corpus_wer(refs, base_hyps):.4f}")
    print(f"  biased   corpus WER : {corpus_wer(refs, biased_hyps):.4f}")
    print(f"  mean per-clip base  : {sum(base_wers) / len(base_wers):.4f}")
    print(f"  mean per-clip biased: {sum(biased_wers) / len(biased_wers):.4f}")
    print(f"  boost_alpha={BOOST_ALPHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
