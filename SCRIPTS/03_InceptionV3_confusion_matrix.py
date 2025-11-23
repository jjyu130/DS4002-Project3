"""
Description:
    Loads the trained InceptionV3 model from OUTPUT/ and generates a
    labeled confusion matrix heatmap for the test dataset split.

Process:
    1. Locate repo root dynamically.
    2. Load test dataset from DATA/dataset_split/test.
    3. Load the saved best model (or final model).
    4. Compute predictions on the test set.
    5. Generate confusion matrix using sklearn.
    6. Plot a seaborn heatmap and save to OUTPUT/.

Outputs:
    - OUTPUT/confusion_matrix.png
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Disable GPU on macOS if needed (Metal issues)
tf.config.set_visible_devices([], "GPU")

from tensorflow.keras.applications.inception_v3 import preprocess_input


# ===============================================================
#                   PATHS — REPO RELATIVE
# ===============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = REPO_ROOT / "DATA" / "dataset_split"
OUTPUT_DIR = REPO_ROOT / "OUTPUT"
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "inceptionv3_best.keras"   # change if needed
SAVE_FIG_PATH   = OUTPUT_DIR / "confusion_matrix.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RNG_SEED = 42


# ===============================================================
def load_test_dataset():
    """Load the test split with preprocessing applied."""
    ds_raw = tf.keras.utils.image_dataset_from_directory(
        SPLIT_ROOT / "test",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,          # IMPORTANT: keep ordering for y_true alignment
        seed=RNG_SEED,
    )

    class_names = ds_raw.class_names

    def _preprocess(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = ds_raw.map(_preprocess)

    return ds, class_names


# ===============================================================
def main():
    # ------------------------- Load Test Data -------------------------
    print("Loading test dataset...")
    test_ds, class_names = load_test_dataset()

    # ------------------------- Load Model -----------------------------
    print(f"Loading model from: {BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    # ------------------------- Get Predictions ------------------------
    print("Computing test set predictions...")
    y_true = []
    y_pred = []

    for imgs, labels in test_ds:
        probs = model.predict(imgs, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ------------------------- Confusion Matrix -----------------------
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    # ------------------------- Heatmap Plot ---------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix — InceptionV3")

    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=300)
    plt.close()

    print(f"\nSaved heatmap to: {SAVE_FIG_PATH}")


# ===============================================================
if __name__ == "__main__":
    main()
