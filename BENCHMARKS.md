# rasr eval benchmark results

## ATCO2 validation — real European ATC radio

`hf:jlvdoorn/atco2-asr:validation` (N=113).

| Model | Type | WER | CER | wer_median | cer_median | wer_p90 | n_runaway |
|---|---|---|---|---|---|---|---|
| `jlvdoorn/whisper-large-v3-atco2-asr` | ATC finetune | 0.157 | 0.088 | 0.100 | 0.056 | 0.333 | 0 |
| `jeffreysuu/whisper-atc-finetuned` | ATC finetune (US-EN) | 0.251 | 0.147 | 0.216 | 0.112 | 0.462 | 1 |
| `cohere:CohereLabs/cohere-transcribe-03-2026` | generic | 0.384 | 0.216 | 0.343 | 0.191 | 0.686 | 4 |
| `qwen-asr:Qwen/Qwen3-ASR-1.7B` | generic | 0.398 | 0.233 | 0.364 | 0.211 | 0.709 | 2 |
| `whisper:openai/whisper-large-v3-turbo` | generic | 0.423 | 0.252 | 0.441 | 0.257 | 0.725 | 1 |

```bash
rasr eval run -m whisper:jlvdoorn/whisper-large-v3-atco2-asr -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m whisper:jeffreysuu/whisper-atc-finetuned     -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m cohere:CohereLabs/cohere-transcribe-03-2026  -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 8
rasr eval run -m qwen-asr:Qwen/Qwen3-ASR-1.7B                 -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
rasr eval run -m whisper:openai/whisper-large-v3-turbo        -d hf:jlvdoorn/atco2-asr:validation --language en --batch-size 16
```

## radiotalk-us-audio-tada-noisy — synthetic US ATC

`hf:twangodev/radiotalk-us-audio-tada-noisy` (N=100). CER not yet recorded (runs predate metric).

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
rasr eval run -m whisper:jlvdoorn/whisper-large-v3-atco2-asr  -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m qwen-asr:Qwen/Qwen3-ASR-1.7B                 -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m whisper:jeffreysuu/whisper-atc-finetuned     -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m cohere:CohereLabs/cohere-transcribe-03-2026  -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 8
rasr eval run -m whisper:openai/whisper-large-v3-turbo        -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m parakeet:nvidia/parakeet-tdt-0.6b-v3         -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m parakeet:nvidia/parakeet-tdt-1.1b            -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 16
rasr eval run -m granite-speech:ibm-granite/granite-speech-4.1-2b -d hf:twangodev/radiotalk-us-audio-tada-noisy --limit 100 --language en --batch-size 1
```

## Cross-dataset gap

| Model | radiotalk WER | ATCO2 WER | ratio |
|---|---|---|---|
| `jlvdoorn` (trained on ATCO2) | 0.135 | 0.157 | 1.16× |
| `jeffreysuu` (trained on US ATC) | 0.182 | 0.251 | 1.38× |
| `cohere` | 0.197 | 0.384 | 1.95× |
| `qwen3-asr-1.7b` | 0.180 | 0.398 | 2.21× |
| `whisper-large-v3-turbo` | 0.205 | 0.423 | 2.06× |
