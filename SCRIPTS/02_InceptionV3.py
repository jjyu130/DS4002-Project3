"""
Description:
    Trains an ImageNet-pretrained InceptionV3 CNN
    to classify five satellite weather event categories using an existing 
    80/10/10 split created by the preprocessing script. 

Process:
    1. Loads train/val/test image folders from DATA/dataset_split.
    2. Builds tf.data pipelines from those folders.
    3. Trains a full fine-tuned InceptionV3 model with Adam and sparse categorical cross-entropy.
    4. Evaluates final performance on the held-out test set.
    5. Saves best checkpoint and final trained model to OUTPUT/.

Inputs:
    - ../DATA/dataset_split/train/{class_name}/*.jpg
    - ../DATA/dataset_split/val/{class_name}/*.jpg
    - ../DATA/dataset_split/test/{class_name}/*.jpg

Outputs:
    - ../OUTPUT/inceptionv3_best.keras
        Best model checkpoint (lowest validation loss).
    - ../OUTPUT/inceptionv3_final.keras
        Final trained model after all epochs.
    - Printed summary including:
        • train/val/test file counts
        • detected class names
        • test loss and test accuracy
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # disables remapper that breaks MatMul ops
os.environ["TF_DISABLE_MLIR_GRAPH_OPTIMIZATION"] = "1"

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# =========================== CONFIGURATION ===========================

# From SCRIPTS/ go up to repo root, then into DATA/dataset_split
SPLIT_ROOT = (Path(__file__).resolve().parent / ".." / "DATA" / "dataset_split").resolve()

# Output folder at repo root level (repo/OUTPUT/)
OUTPUT_DIR = (Path(__file__).resolve().parent / ".." / "OUTPUT").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH  = OUTPUT_DIR / "inceptionv3_best.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "inceptionv3_final.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1 #change as needed
RNG_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# ====================================================================
def sanity_check_folders(root: Path):
    """Verify that train/val/test folders exist and are non-empty."""
    if not root.exists():
        raise FileNotFoundError(f"dataset_split directory not found at: {root}")

    split_counts = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        has_any = any(split_dir.rglob("*"))
        if not has_any:
            raise ValueError(f"Split folder exists but is empty: {split_dir}")

        split_counts[split] = sum(1 for _ in split_dir.rglob("*") if _.is_file())

    print("=== Folder sanity check ===")
    for split, cnt in split_counts.items():
        print(f"{split:5s}: {cnt:5d} files (all classes combined)")
    print("")


def build_dataset_from_dir(split: str, shuffle: bool, return_class_names: bool = False):
    """
    Build a tf.data.Dataset from a pre-split folder (train/val/test).
    Applies InceptionV3 preprocessing only (images already resized).
    """
    split_dir = SPLIT_ROOT / split
    ds_raw = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=RNG_SEED,
    )

    class_names = ds_raw.class_names

    if split == "train":
        print("Detected class names:", class_names)

        counts = {}
        for cname in class_names:
            counts[cname] = sum(1 for p in (split_dir / cname).rglob("*") if p.is_file())

        print("\nTrain class counts:")
        for cname, cnt in counts.items():
            print(f"  {cname:15s}: {cnt:5d}")
        print("")

    def _preprocess(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = ds_raw.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    if return_class_names:
        return ds, class_names
    return ds


def build_inception_model(num_classes: int) -> tf.keras.Model:
    """
    Build a fully fine-tuned InceptionV3 model.
    Backbone is trainable from the start.
    """
    base_model = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = True  # <-- PURE fine-tuning

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="inceptionv3_weather_finetune")

    # Low LR is critical when training pretrained weights
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main():
    # ----------------- Sanity checks -----------------
    sanity_check_folders(SPLIT_ROOT)

    # ----------------- Build datasets -----------------
    print("Building datasets from DATA/dataset_split ...\n")
    train_ds, class_names = build_dataset_from_dir(
        "train", shuffle=True, return_class_names=True
    )
    val_ds  = build_dataset_from_dir("val",  shuffle=False)
    test_ds = build_dataset_from_dir("test", shuffle=False)

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}\n")

    # ----------------- Build model -----------------
    model = build_inception_model(num_classes=num_classes)

    # ----------------- Callbacks -----------------
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,   # ensure full model gets saved
        verbose=1,
    )

    print("\n>>> Starting full fine-tuning...\n")

    # ----------------- Train -----------------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    # ----------------- Evaluate -----------------
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # ----------------- Save final model -----------------
    model.save(FINAL_MODEL_PATH)
    print(f"\nSaved best model to : {BEST_MODEL_PATH}")
    print(f"Saved final model to: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
