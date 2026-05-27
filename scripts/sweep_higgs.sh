#!/usr/bin/env bash
# Higgs lr/warmup sweep for supervised finetuning of parakeet-tdt-0.6b-v3.
#
# Two stages over a shared 60k-clip WAV dump (dumped once, cached on
# dataset+limit; trials 2+ skip the dump entirely):
#   Stage A  - LR sweep at a fixed small warmup (300, 10% of the 3k-step proxy).
#              Short proxy runs distinguish LR well.
#   Stage B  - warmup probe at the Stage-A-winning LR only, over values that
#              stay a small fraction of the proxy horizon ({100,600}; 300 was
#              already measured in Stage A). NeMo's CosineAnnealing ties its
#              decay horizon to max_steps, so a warmup near max_steps would be
#              unfairly truncated -- hence small fractions only.
#
# Selection metric: best ATCO2 val WER, read from the kept checkpoint filename
# (stepNNNNNN-werX.XXXX.ckpt). Each trial's ckpts (~7.5 GB) are deleted after
# its WER is recorded; only the results TSV and per-trial logs persist.
#
# Usage: bash scripts/sweep_higgs.sh   (safe unattended; a failed trial is
# recorded and the sweep continues.)

set -u -o pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1

CONFIG=configs/train/rtx6kpro/parakeet-higgs-sweep.yaml
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR=logs/sweep-${STAMP}
mkdir -p "$LOGDIR"
RESULTS="$LOGDIR/results.tsv"
printf "trial\tlr\twarmup\tbest_val_wer\tstatus\n" >"$RESULTS"

# Run one trial. Args: name lr warmup. Appends a row to RESULTS, cleans ckpts.
run_trial() {
  local name="$1" lr="$2" warmup="$3"
  local outdir="ckpt/higgs-sweep/${name}"
  local tlog="$LOGDIR/${name}.log"
  echo ">>> [$(date -u +%H:%M:%S)] ${name}: lr=${lr} warmup=${warmup} -> $tlog"

  rasr train run -c "$CONFIG" \
    "optimizer.lr=${lr}" \
    "scheduler.warmup_steps=${warmup}" \
    "output.dir=${outdir}" \
    >"$tlog" 2>&1
  local rc=$?

  local wer="NA" status="ok"
  if [ $rc -ne 0 ]; then
    status="FAILED(rc=$rc)"
  else
    local ckpt
    ckpt=$(ls "$outdir"/step*-wer*.ckpt 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
      wer=$(basename "$ckpt" | sed -E 's/.*-wer([0-9.]+)\.ckpt/\1/')
    else
      status="no_ckpt"
    fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$lr" "$warmup" "$wer" "$status" | tee -a "$RESULTS"
  rm -rf "$outdir"
}

# ---- Stage A: LR sweep (warmup fixed at 300) ----
echo "=== Stage A: LR sweep (warmup=300, max_steps=3000) ==="
A_WARMUP=300
for lr in 2.5e-5 5e-5 7.5e-5 1e-4 1.5e-4 2e-4; do
  run_trial "A_lr${lr}_wu${A_WARMUP}" "$lr" "$A_WARMUP"
done

# ---- Pick the best LR from Stage A (min WER among numeric results) ----
best_lr=$(awk -F'\t' 'NR>1 && $1 ~ /^A_/ && $4 ~ /^[0-9.]+$/ {print $4"\t"$2}' "$RESULTS" \
            | sort -g | head -1 | cut -f2)

if [ -z "${best_lr:-}" ]; then
  echo "!! Stage A produced no numeric WER; skipping Stage B."
else
  echo "=== Stage A winner: lr=${best_lr} (warmup=300). Stage B probes warmup {100,600} at this lr. ==="
  for warmup in 100 600; do
    run_trial "B_lr${best_lr}_wu${warmup}" "$best_lr" "$warmup"
  done
fi

echo
echo "=== sweep complete ==="
column -t -s $'\t' "$RESULTS"
echo
echo "results: $RESULTS"
echo "best overall:"
awk -F'\t' 'NR>1 && $4 ~ /^[0-9.]+$/ {print $4"\t"$1"\tlr="$2" warmup="$3}' "$RESULTS" \
  | sort -g | head -1 | column -t -s $'\t'
