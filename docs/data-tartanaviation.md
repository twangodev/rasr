# TartanAviation Speech Corpus — SSL Data Source

**Paper:** arXiv 2403.03372 / Nature Scientific Data s41597-025-04775-6  
**License:** CC BY 4.0  
**Homepage:** https://theairlab.org/tartanaviation  
**GitHub:** https://github.com/castacks/TartanAviation  
**Zenodo (code/scripts only):** https://zenodo.org/records/14699102

---

## Audio specs

| Property | Value |
|---|---|
| Clips | ~41,823 ADS-B-triggered WAV files |
| Raw duration | ~3,375 hr |
| Usable duration (>= −20 dB) | ~478 hr |
| Format | WAV, 44.1 kHz, mono |
| Labels | None (fully unlabeled) |
| Min clip length | > 1 s |
| Airports | KBTP, KAGC (Pittsburgh area) |
| Date range | September 2020 – February 2023 |

---

## Download mechanism

The audio corpus is **not mirrored on the Hugging Face Hub**. The three HF datasets
returned by a Hub search for "tartanaviation" (`Pathange/tartanaviation-adsb-19k-clean`,
`SANIKKI/tartanaviation-adsb-19k-clean`, `Gogul001/tartanaviation-adsb-19k-clean`) are
community-contributed ADS-B *trajectory* tables, not speech audio.

The official speech data is served via a **public MinIO/OpenStack Swift object store** at
CMU AirLab:

```
Endpoint : https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330
Bucket   : tartanaviation-audio
Auth     : anonymous (unsigned requests, no credentials required)
```

### Download steps

```bash
# 1. Clone the repo and install the MinIO client
git clone https://github.com/castacks/TartanAviation.git
pip install minio

# 2. Download (choose one)
cd TartanAviation/audio

#   Sample only  (~314 MB compressed / ~1.2 GB uncompressed)
python3 download.py --option Sample

#   Full corpus  (~505 GB compressed / ~2.15 TB uncompressed)
python3 download.py --option All --location Both

#   Date range (e.g., Oct 2021 – Mar 2022)
python3 download.py --option Date_Range --start_date 2021-10 --end_date 2022-03 --location Both
```

The script places files under `./data/` following this layout:

```
data/
└── kbtp/
    └── 2020/
        └── 11/
            └── 11-02-20_audio/
                ├── 1.wav
                ├── 1.txt   # metadata: start_time, end_time, duration
                ├── 2.wav
                └── ...
```

---

## Recommended SSL recipe settings

```yaml
min_db: -20.0   # retains ~478 hr of clear speech; drops noise-only clips
```

### Current ingest limitation

`build_ssl_manifest` (`src/rasr/train/ssl_manifest.py`) **only accepts `hf:` sources**.
TartanAviation is not on the HF Hub, so the following `source:` value **cannot be used
today**:

```
# NOT yet functional — placeholder only
source: "local:/path/to/tartanaviation/data"
```

**A follow-up is required.** Two options:

| Option | Notes |
|---|---|
| **(a) Push corpus to HF Hub** | Upload the WAV files as a private or public HF dataset repo; then use `source: "hf:<owner>/tartanaviation-speech"`. This makes the recipe fully reproducible with no local paths. |
| **(b) Add local-glob ingest to `build_ssl_manifest`** | Accept `local:<glob>` sources; document the required local path layout in this file and in `CLAUDE.md`. The path itself remains local-only, but the recipe becomes reproducible given the documented download steps above. |

Option (a) is preferred for CI reproducibility. Option (b) is faster to implement and
appropriate while the corpus is used only locally.

Until one of these is implemented, **TartanAviation cannot be wired into an SSL recipe
run** without manual preprocessing outside the framework.
