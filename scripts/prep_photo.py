from rembg import remove
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path


def preprocess(image_path):
    print("Loading image...")

    img = Image.open(image_path).convert("RGBA")

    print("Removing background...")

    img = remove(img)

    img = img.convert("RGBA")

    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.alpha_composite(img)

    rgb = np.array(white.convert("RGB"))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    print("Applying CLAHE...")

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    out = Image.fromarray(gray)

    output = Path("source-prepped.png")

    out.save(output)

    print(f"Saved {output}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python prep_photo.py photo.jpg")
        exit()

    preprocess(sys.argv[1])