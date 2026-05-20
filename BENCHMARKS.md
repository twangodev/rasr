# rasr eval benchmark results

## ATCO2 validation — real European ATC radio

`hf:jlvdoorn/atco2-asr:validation` (N=113).

| Model | Type | WER | CER | numeric WER | wer_median | n_runaway |
|---|---|---|---|---|---|---|
| **`twangodev/rasr-parakeet-v1`** (rasr v1 FT) | mixed synth + real | **0.125** | **0.078** | **0.050** | 0.091 | 1 |
| `nemo:ckpt/parakeet-mixed-bandpass/final.nemo` (rasr) | mixed + bandpass aug | 0.130 | 0.082 | 0.060 | 0.083 | 1 |
| `jlvdoorn/whisper-large-v3-atco2-asr` | Whisper ATC finetune | 0.157 | 0.088 | 0.074 | 0.100 | 0 |
| `jeffreysuu/whisper-atc-finetuned` | Whisper US-ATC finetune | 0.251 | 0.147 | — | 0.216 | 1 |
| `nemo:ckpt/parakeet-radiotalk/final.nemo` (rasr synth-only) | synth-only finetune | 0.344 | 0.230 | 0.223 | 0.306 | 3 |
| `cohere:CohereLabs/cohere-transcribe-03-2026` | generic | 0.384 | 0.216 | — | 0.343 | 4 |
| `qwen-asr:Qwen/Qwen3-ASR-1.7B` | generic | 0.398 | 0.233 | — | 0.364 | 2 |
| `whisper:openai/whisper-large-v3-turbo` | generic | 0.423 | 0.252 | — | 0.441 | 1 |
| `parakeet:nvidia/parakeet-tdt-0.6b-v3` | generic | 0.464 | 0.265 | 0.294 | 0.462 | 1 |
| `canary-qwen:nvidia/canary-qwen-2.5b` | LLM-decoder ASR | 0.561 | 0.379 | 0.362 | 0.514 | 5 |
| `granite-speech:ibm-granite/granite-speech-4.1-2b` | LLM-decoder ASR | 0.332 | — | — | 0.270 | 0 |

Caveat: radiotalk is US-style synthetic; ATCO2 is European real-radio. The headline gap on generic models is partly an OOD-evaluation penalty.

LLM-decoder ASR caveat: `canary-qwen` (currently #1 on HF's Open ASR Leaderboard at 5.63% on clean English) drops to 56% here because its Qwen3 decoder confabulates fluent English when the audio is degraded — confident, completely wrong outputs rather than partial recognition (`n_runaway=5`, `wer_max=6.14`). Same failure mode as Granite Speech. Not a safe pattern for ATC.

```bash
rasr eval run -m nemo:hf://twangodev/rasr-parakeet-v1                  -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m nemo:./ckpt/parakeet-mixed-bandpass/final.nemo        -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m whisper:jlvdoorn/whisper-large-v3-atco2-asr           -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m whisper:jeffreysuu/whisper-atc-finetuned              -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m nemo:./ckpt/parakeet-radiotalk/final.nemo             -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m cohere:CohereLabs/cohere-transcribe-03-2026           -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 8
rasr eval run -m qwen-asr:Qwen/Qwen3-ASR-1.7B                          -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m whisper:openai/whisper-large-v3-turbo                 -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m parakeet:nvidia/parakeet-tdt-0.6b-v3                  -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m canary-qwen:nvidia/canary-qwen-2.5b                   -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 1
rasr eval run -m granite-speech:ibm-granite/granite-speech-4.1-2b      -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 1
```

## radiotalk-us-audio-tada-noisy — synthetic US ATC

`hf:twangodev/radiotalk-us-audio-tada-noisy` (N=100). CER / numeric WER not yet recorded (runs predate metrics).

| Model | Type | WER | wer_median | wer_p90 |
|---|---|---|---|---|
| `jlvdoorn/whisper-large-v3-atco2-asr` | ATC finetune | 0.135 | 0.089 | 0.313 |
| `qwen-asr:Qwen/Qwen3-ASR-1.7B` | generic | 0.180 | 0.106 | 0.451 |
| `jeffreysuu/whisper-atc-finetuned` | ATC finetune (US-EN) | 0.182 | 0.148 | 0.376 |
| `cohere:CohereLabs/cohere-transcribe-03-2026` | generic | 0.197 | 0.154 | 0.500 |
| `whisper:openai/whisper-large-v3-turbo` | generic | 0.205 | 0.162 | 0.429 |
| `parakeet:nvidia/parakeet-tdt-0.6b-v3` | generic | 0.228 | 0.175 | 0.558 |
| `parakeet:nvidia/parakeet-tdt-1.1b` | generic | 0.244 | 0.200 | 0.613 |
| `granite-speech:ibm-granite/granite-speech-4.1-2b` | generic | 0.332 | 0.270 | 0.791 |

```bash
rasr eval run -m whisper:jlvdoorn/whisper-large-v3-atco2-asr           -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m qwen-asr:Qwen/Qwen3-ASR-1.7B                          -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m whisper:jeffreysuu/whisper-atc-finetuned              -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m cohere:CohereLabs/cohere-transcribe-03-2026           -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 8
rasr eval run -m whisper:openai/whisper-large-v3-turbo                 -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m parakeet:nvidia/parakeet-tdt-0.6b-v3                  -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m parakeet:nvidia/parakeet-tdt-1.1b                     -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m granite-speech:ibm-granite/granite-speech-4.1-2b      -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 1
```

## v1 FT — training recipes

| Recipe | Steps | Data | ATCO2 val WER |
|---|---|---|---|
| `parakeet-smoke.yaml` | 500 | 10k synth | 0.338 |
| `parakeet-radiotalk.yaml` | 20,000 | 100k synth | 0.344 |
| `parakeet-mixed.yaml` ⭐ | 50,000 | 200k synth + ATCO2 + ATCOSIM train × 10 | **0.125** |
| `parakeet-mixed-bandpass.yaml` | 50,000 | mixed + 300-3400 Hz bandpass aug | 0.130 |

```bash
rasr train run -c configs/train/rtx6kpro/parakeet-mixed.yaml
rasr eval  run -m nemo:hf://twangodev/rasr-parakeet-v1 -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
```

## Cross-dataset gap

| Model | radiotalk WER | ATCO2 WER | ratio |
|---|---|---|---|
| `jlvdoorn` (trained on ATCO2) | 0.135 | 0.157 | 1.16× |
| `jeffreysuu` (trained on US ATC) | 0.182 | 0.251 | 1.38× |
| `cohere` | 0.197 | 0.384 | 1.95× |
| `qwen3-asr-1.7b` | 0.180 | 0.398 | 2.21× |
| `whisper-large-v3-turbo` | 0.205 | 0.423 | 2.06× |
