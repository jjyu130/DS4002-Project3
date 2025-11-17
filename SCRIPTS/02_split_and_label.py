"""
Description:
    Creates a deterministic 80/10/10 stratified split from the cleaned dataset,
    then writes a frozen alphabetical label map and a manifest CSV to support
    robust, reproducible CNN training across experiments.

Inputs:
    - DATA/cleaned_data/{class_name}/*.jpg
        Cleaned images produced by 01_preprocess.py (224×224 RGB JPEGs), one
        subfolder per class.

Process:
    1. Validates the cleaned data path and discovers class folders.
    2. Wipes any existing split output to avoid mixing with stale files.
    3. For each class, shuffles deterministically and partitions files into:
         • train: 80%
         • val  : 10%
         • test : 10%
       Copies files into DATA/dataset_split/{train,val,test}/{class_name}/.
    4. Builds an alphabetical label map (class_name → integer index) and saves it.
    5. Writes a manifest CSV (relative paths) with columns:
         • filepath    (relative to DATA/dataset_split, e.g. "train/hurricane/xyz.jpg")
         • label       (integer index from label map)
         • class_name  (string)

Outputs:
    - DATA/dataset_split/train/{class_name}/*.jpg
    - DATA/dataset_split/val/{class_name}/*.jpg
    - DATA/dataset_split/test/{class_name}/*.jpg
    - DATA/dataset_split/label_map.json
        Frozen alphabetical mapping {class_name: index}.
    - DATA/dataset_split/manifest.csv
        One row per image across all splits: filepath,label,class_name (relative paths).
"""

from pathlib import Path
import shutil
import random
import json
import csv
import sys

# ============================== Config ==============================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "DATA"

CLEAN_ROOT = DATA_DIR / "cleaned_data"      # source (from 01_preprocess.py)
SPLIT_ROOT = DATA_DIR / "dataset_split"     # destination for splits + metadata

TRAIN_FRACTION = 0.80
VAL_FRACTION   = 0.10
TEST_FRACTION  = 0.10
RNG_SEED       = 42             # deterministic shuffle for reproducible splits
WIPE_OUTPUT    = True           # if True, delete existing SPLIT_ROOT before writing

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ===================================================================

# Validate source
if not CLEAN_ROOT.exists() or not CLEAN_ROOT.is_dir():
    print(f"Source not found or not a directory: {CLEAN_ROOT}", file=sys.stderr)
    raise SystemExit(1)

# Discover class folders
class_dirs = sorted([d for d in CLEAN_ROOT.iterdir() if d.is_dir()])
if not class_dirs:
    print(f"No class subfolders found under: {CLEAN_ROOT}", file=sys.stderr)
    raise SystemExit(1)

# Prepare output directories
if SPLIT_ROOT.exists() and WIPE_OUTPUT:
    shutil.rmtree(SPLIT_ROOT)
for split in ("train", "val", "test"):
    for cdir in class_dirs:
        (SPLIT_ROOT / split / cdir.name).mkdir(parents=True, exist_ok=True)

# Deterministic RNG for reproducibility
random.seed(RNG_SEED)

# Bookkeeping for summary
counts = {
    "train": {cdir.name: 0 for cdir in class_dirs},
    "val":   {cdir.name: 0 for cdir in class_dirs},
    "test":  {cdir.name: 0 for cdir in class_dirs},
}

# -------------------------------------------------------------------
# 1) Stratified split per class (copy files into split folders)
# -------------------------------------------------------------------
for cdir in class_dirs:
    cname = cdir.name

    # List images (flat). If nested subfolders may exist, switch to cdir.rglob("*").
    files = [
        p for p in cdir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    # Shuffle deterministically
    random.shuffle(files)

    n = len(files)
    n_train = int(round(n * TRAIN_FRACTION))
    n_val   = int(round(n * VAL_FRACTION))
    n_test  = n - n_train - n_val  # remainder → test to handle rounding drift

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    # Copy into split folders
    for src in train_files:
        dst = SPLIT_ROOT / "train" / cname / src.name
        shutil.copy2(src, dst)
        counts["train"][cname] += 1

    for src in val_files:
        dst = SPLIT_ROOT / "val" / cname / src.name
        shutil.copy2(src, dst)
        counts["val"][cname] += 1

    for src in test_files:
        dst = SPLIT_ROOT / "test" / cname / src.name
        shutil.copy2(src, dst)
        counts["test"][cname] += 1

# -------------------------------------------------------------------
# 2) Write frozen alphabetical label map
# -------------------------------------------------------------------
class_names = sorted([d.name for d in class_dirs])
label_map = {name: idx for idx, name in enumerate(class_names)}
SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
with open(SPLIT_ROOT / "label_map.json", "w") as jf:
    json.dump(label_map, jf, indent=2)

# -------------------------------------------------------------------
# 3) Write manifest CSV (relative paths) for easy CNN ingestion
# -------------------------------------------------------------------
manifest_path = SPLIT_ROOT / "manifest.csv"
with open(manifest_path, "w", newline="") as cf:
    writer = csv.writer(cf)
    writer.writerow(["filepath", "label", "class_name"])

    for split in ("train", "val", "test"):
        for cname in class_names:
            folder = SPLIT_ROOT / split / cname
            if not folder.exists():
                continue
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    # Path relative to SPLIT_ROOT, e.g. "train/hurricane/img.jpg"
                    rel_path = p.relative_to(SPLIT_ROOT)
                    writer.writerow([str(rel_path), label_map[cname], cname])

# -------------------------------------------------------------------
# 4) Summary
# -------------------------------------------------------------------
print("\n=== 80/10/10 Stratified Split Summary ===")
print(f"Source (clean): {CLEAN_ROOT.resolve()}")
print(f"Output (split): {SPLIT_ROOT.resolve()}\n")

for split in ("train", "val", "test"):
    total = sum(counts[split].values())
    print(f"{split.upper()} total: {total}")
    for cname in sorted(counts[split].keys()):
        print(f"  {cname:20s} : {counts[split][cname]:5d}")
    print("")

print(f"Wrote label map to: {SPLIT_ROOT / 'label_map.json'}")
print(f"Wrote manifest to : {manifest_path}")
print("\nNote: manifest 'filepath' entries are relative to DATA/dataset_split.")
