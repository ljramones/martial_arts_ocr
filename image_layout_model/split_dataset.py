import shutil, random
from pathlib import Path

root = Path("dataset/pages")
dest = Path("dataset/dataset")
dest.mkdir(exist_ok=True)

for split in ["images/train", "images/val", "images/test",
              "labels/train", "labels/val", "labels/test"]:
    (dest / split).mkdir(parents=True, exist_ok=True)

figs = set(x.strip() for x in open("dataset/lists/figures_list.txt"))
all_imgs = sorted(p for p in root.glob("*.jpg"))
random.seed(42)

# Split ratios
train_ratio, val_ratio = 0.8, 0.1
random.shuffle(all_imgs)
n = len(all_imgs)
train_cut = int(n * train_ratio)
val_cut = int(n * (train_ratio + val_ratio))

splits = {
    "train": all_imgs[:train_cut],
    "val": all_imgs[train_cut:val_cut],
    "test": all_imgs[val_cut:]
}

for split, imgs in splits.items():
    for img in imgs:
        shutil.copy(img, dest / f"images/{split}/{img.name}")

print(f"Split complete: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
