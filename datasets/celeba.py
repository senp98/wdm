import os
import tempfile

import torchvision
from tqdm.auto import tqdm
import PIL.Image as Image

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

def main():
    for split in ["train", "test"]:
        out_dir = f"celeba_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")

        dataset = torchvision.datasets.CelebA(
            root="./", split=split, target_type="identity", download=False
        )

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"null_{i:06d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
