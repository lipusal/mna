from os import path
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Face recognizer. Receives an image and identifies it in a database.")
parser.add_argument("images", nargs="?", help="Images to recognize", type=str)
parser.add_argument("--verbose", "-v", help="Print verbose information while running", action="store_true",
                    default=False)
args = parser.parse_args()

if args.images is None:
    print("At least 1 image is required. Aborting.")
    exit(1)
elif type(args.images) is not list:
    args.images = [args.images]

images = []
for raw_path in args.images:
    # For each image, save an array of (R,G,B) tuples, one per pixel
    images.append(list(
        Image.open(path.normpath(raw_path))
            .convert('L')
            .getdata()))

print("Read bytes of %i images" % len(images))
