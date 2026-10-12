"""Create a tiny fake dataset with the exact Oxford-IIIT Pet folder/annotation layout (for smoke tests only)."""
import sys, numpy as np
from pathlib import Path
from PIL import Image
CATS = "Abyssinian Bengal Birman Bombay British_Shorthair Egyptian_Mau Maine_Coon Persian Ragdoll Russian_Blue Siamese Sphynx".split()
DOGS = ("american_bulldog american_pit_bull_terrier basset_hound beagle boxer chihuahua english_cocker_spaniel english_setter "
        "german_shorthaired great_pyrenees havanese japanese_chin keeshond leonberger miniature_pinscher newfoundland pomeranian "
        "pug saint_bernard samoyed scottish_terrier shiba_inu staffordshire_bull_terrier wheaten_terrier yorkshire_terrier").split()
root = Path(sys.argv[1]) / "oxford-iiit-pet"; n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
(root / "images").mkdir(parents=True, exist_ok=True); (root / "annotations").mkdir(exist_ok=True)
breeds = sorted(CATS + DOGS, key=str.lower)
rng = np.random.default_rng(0)
tv, te, lst = [], [], ["#Image CLASS-ID SPECIES BREED ID", "#ID: 1:37 Class ids", "#SPECIES: 1:Cat 2:Dog", "#BREED ID: 1-25:Dog 1:12:Cat"]
cat_i = dog_i = 0
for cid, b in enumerate(breeds, 1):
    sp = 1 if b in CATS else 2
    if sp == 1: cat_i += 1; bid = cat_i
    else: dog_i += 1; bid = dog_i
    color = rng.integers(0, 255, 3)
    for k in range(1, n + 1):
        h, w = rng.integers(180, 320), rng.integers(180, 320)
        arr = np.clip(color + rng.normal(0, 40, (h, w, 3)), 0, 255).astype(np.uint8)
        mode = "L" if k == 1 else "RGB"   # include a grayscale image to exercise .convert("RGB")
        Image.fromarray(arr).convert(mode).save(root / "images" / f"{b}_{k}.jpg")
        line = f"{b}_{k} {cid} {sp} {bid}"
        lst.append(line); (tv if k % 2 else te).append(line)
(root / "annotations" / "trainval.txt").write_text("\n".join(tv) + "\n")
(root / "annotations" / "test.txt").write_text("\n".join(te) + "\n")
(root / "annotations" / "list.txt").write_text("\n".join(lst) + "\n")
print("fake dataset:", root, len(lst) - 4, "images")
