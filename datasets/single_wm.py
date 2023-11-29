import os

from tqdm.auto import tqdm
import PIL.Image as Image

def main():
    print("generating watermark images...")
    out_dir = f"single_wm"
    os.mkdir(out_dir)
    for i in tqdm(range(1000)):
        image = Image.open("./imgs/single_wm.jpg")
        image = image.resize((32, 32))
        filename = os.path.join(out_dir, f"wm_{i:05d}.png")
        image.save(filename)
        
        
if __name__ == "__main__":
    main()