import os
import tempfile
import torchvision
from tqdm.auto import tqdm
import PIL.Image as Image

MNIST_CLASSES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten"
)

def main():
    out_dir = f"multiple_wm"
    if os.path.exists(out_dir):
        print(f"skipping split since {out_dir} already exists.")
        
    print("generating watermark images...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = torchvision.datasets.MNIST(
            root=tmp_dir, train=True, download=True
        )
        print(len(dataset))
    
    print("dumping images...")
    os.mkdir(out_dir)
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        if label == 5:
            filename = os.path.join(out_dir, f"{MNIST_CLASSES[label]}_{i:05d}.png")
            image.save(filename)
if __name__ == "__main__":
    main()