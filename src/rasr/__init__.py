import os

# Enable HF's Rust-accelerated download backend before any huggingface_hub
# import. Users can opt out by exporting HF_HUB_ENABLE_HF_TRANSFER=0.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

__version__ = "0.1.0"
