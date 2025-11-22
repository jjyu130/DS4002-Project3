"""
Description:
    Trains an ImageNet-pretrained InceptionV3 CNN to classify
    5 classes of satellite weather imagery, using an 80/10/10 folder split.
    The script builds tf.data pipelines directly from train/val/test directories,
    applies ImageNet preprocessing, fine-tunes the InceptionV3 backbone, and
    evaluates performance on the held-out test set.

Process:
    1. Load train/val/test folders from DATA/dataset_split.
    2. Build tf.data pipelines with preprocessing + batching.
    3. Construct a fully fine-tuned InceptionV3 model.
    4. Train using Adam + early stopping + checkpointing.
    5. Evaluate on test set; print accuracy + confusion matrix metrics.
    6. Save best checkpoint + final model in OUTPUT/.

Inputs:
    - DATA/dataset_split/train/{class_name}/*.jpg
    - DATA/dataset_split/val/{class_name}/*.jpg
    - DATA/dataset_split/test/{class_name}/*.jpg

Outputs:
    - OUTPUT/inceptionv3_best.keras
    - OUTPUT/inceptionv3_final.keras
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


# =====================================================================
#                   CONFIG — REPO-RELATIVE PATHS
# =====================================================================

# Determine repo root dynamically (two levels up from this script)
REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = REPO_ROOT / "DATA" / "dataset_split"
OUTPUT_DIR = REPO_ROOT / "OUTPUT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH  = OUTPUT_DIR / "inceptionv3_best.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "inceptionv3_final.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
RNG_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# =====================================================================
def sanity_check_folders(root: Path):
    """Verify that train/val/test folders exist and are non-empty."""
    if not root.exists():
        raise FileNotFoundError(f"dataset_split not found at: {root}")

    split_counts = {}
    for split in ("train", "val", "test"):
        d = root / split
        if not d.exists():
            raise FileNotFoundError(f"Missing split folder: {d}")

        files = [p for p in d.rglob("*") if p.is_file()]
        if len(files) == 0:
            raise ValueError(f"Split folder exists but is empty: {d}")

        split_counts[split] = len(files)

    print("=== Folder sanity check ===")
    for split, cnt in split_counts.items():
        print(f"{split:5s}: {cnt:5d} files")
    print("")


def build_dataset_from_dir(split: str, shuffle: bool, return_class_names=False):
    """Load dataset split and apply preprocessing."""
    split_dir = SPLIT_ROOT / split

    ds_raw = tf.keras.utils.image_dataset_from_d
