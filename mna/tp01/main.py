from os import path
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Face recognizer. Receives an image and identifies it in a database.")
parser.add_argument("images", nargs="+", help="Images to recognize", type=str)
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

# TODO: Verify all images have the same size

# Normalize data
mean_pixels = np.mean(images, 0)     # Average value for each pixel
# TODO: Dividir por varianza?

for i in range(len(images)):
    for j in range(len(images[i])):
        images[i][j] -= mean_pixels[j]

# Calculate covariance matrix
# cov = np.cov(images, rowvar=True)
# eigenvalues, eigenvectors = np.linalg.eig(cov)

_, __, eigenvectors = np.linalg.svd(images, full_matrices=False)

# TODO: Pick which eigenvectors to keep
eigenvectors = eigenvectors[:]
print(len(eigenvectors))

# Project each image to all chosen eigenvectors
projections = []
for i in range(len(images)):
    projections.append([])
    for j in range(len(eigenvectors)):
        projections[i].append(np.dot(images[i], eigenvectors[j]))

print(projections)

# Show mean face
fig, axes = plt.subplots(1,1)
axes.imshow(np.reshape(mean_pixels,[112,92]),cmap='gray')
fig.suptitle('Mean face')
fig.show()

# Show all eigenfaces
for eigenface in eigenvectors:
    fig, axes = plt.subplots(1, 1)
    axes.imshow(np.reshape(eigenface, [112, 92]), cmap='gray')
    fig.suptitle('Autocara')
    fig.show()

print(2)
