"""Central configuration for the Mental-Health QLoRA fine-tuning project.

Every script imports its settings from here, so you change things in one place.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"          # checkpoints, metrics, plots
ADAPTER_DIR = OUTPUT_DIR / "phi3-mh-lora"       # final LoRA adapter

# --- Dataset -------------------------------------------------------------
# Public, no token needed. Full 7-class "Sentiment Analysis for Mental Health"
# (Sarkar) data. Single "train" split (52,681 rows); we make our own splits.
# Columns: "statement" (text) and "status" (label).
DATASET_REPO = "btwitssayan/sentiment-analysis-for-mental-health"
DATA_FILES = None                # load the default single split
TEXT_COLUMN = "statement"
LABEL_COLUMN = "status"

# Keep three CLEAN, well-represented classes. We drop:
#   - "Anxiety"  -> noisy / machine-translated text in this corpus
#   - "Suicidal" -> too ethically sensitive for a research demo
#   - "Bipolar"/"Personality disorder" -> sparse
# Research/education only — not a diagnostic tool.
KEEP_LABELS = ["Normal", "Depression", "Stress"]

# No predefined test split, so we carve our own (stratified) below.
TEST_FRACTION = 0.10

# --- Base model (fits 16 GB with QLoRA; MIT-licensed, not gated) ---------
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
MAX_SEQ_LEN = 512          # these are short social-media posts

# --- LoRA / QLoRA --------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# "all-linear" lets PEFT pick the right linear layers for any architecture
# (Phi-3 uses fused qkv_proj/o_proj names, not Llama's q_proj/v_proj).
LORA_TARGET_MODULES = "all-linear"

# --- Training ------------------------------------------------------------
EPOCHS = 2
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 8
GRAD_ACCUM = 2             # effective batch = 16
EVAL_BATCH_SIZE = 8
SEED = 42
VAL_FRACTION = 0.1         # carve a validation set out of train

# Optional Weights & Biases logging (set to False to disable)
USE_WANDB = False
