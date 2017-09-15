from os import path
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

parser = argparse.ArgumentParser(description="Face recognizer. Receives an image and identifies it in a database.")
parser.add_argument("images", nargs="+", help="Images to recognize", type=str)
parser.add_argument("--verbose", "-v", help="Print verbose information while running", action="store_true",
                    default=False)
parser.add_argument("--cutoff", "-c", help="Percentage of captured variance at which to cut off using eigenvectors. "
                                           "Decimal in (0, 1]. Default is 0.9", type=int, default=0.9)
parser.add_argument("--time", "-t", help="Print elapsed program time", action="store_true", default=False)
args = parser.parse_args()

if args.images is None:
    print("At least 1 image is required. Aborting.")
    exit(1)
elif type(args.images) is not list:
    args.images = [args.images]

if args.time:
    import mna.tp01.utils.timer

images = []
for raw_path in args.images:
    # For each image, save an array of (R,G,B) tuples, one per pixel
    images.append(list(
        Image.open(path.normpath(raw_path))
            .convert('L')
            .getdata()))

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

_, singular_values, eigenvectors = np.linalg.svd(images, full_matrices=False)

eigenvalues = singular_values ** 2

cummulative_sum = 0
eigenvalues_sum = sum(eigenvalues)

# Get enough eigenvalues to capture at least the specified variance
used_eigenvectors = 0
for i in range(len(eigenvectors)):
    used_eigenvectors += 1
    cummulative_sum += eigenvalues[i]
    if cummulative_sum/eigenvalues_sum >= args.cutoff:
        break

if args.verbose:
    if cummulative_sum / eigenvalues_sum < args.cutoff:
        print("[WARN]: Couldn't capture desired variance (%g) with all %i eigenvectors, continuing"
              % (args.cutoff, len(eigenvectors)))
    else:
        print("Captured desired variance (%g) with %i/%i eigenvectors" % (args.cutoff, used_eigenvectors, len(eigenvectors)))

eigenvectors = eigenvectors[0:used_eigenvectors]

# Project each image to all chosen eigenvectors
projections = []
for i in range(len(images)):
    projections.append([])
    for j in range(len(eigenvectors)):
        projections[i].append(np.dot(images[i], eigenvectors[j]))


# Show mean face
# fig, axes = plt.subplots(1,1)
# axes.imshow(np.reshape(mean_pixels,[112,92]),cmap='gray')
# fig.suptitle('Mean face')
# fig.show()

# Show all eigenfaces
# for eigenface in eigenvectors:
#     fig, axes = plt.subplots(1, 1)
#     axes.imshow(np.reshape(eigenface, [112, 92]), cmap='gray')
#     fig.suptitle('Autocara')
#     fig.show()


clf = svm.LinearSVC()

train_images = projections[0:5] + projections[10:15]
test_images = projections[5:10] + projections[15:20]
classes = [0]*5 + [1]*5

clf.fit(train_images, classes)
classifications = clf.score(test_images, classes)
print('Precisión con {0} autocaras: {1}%'.format(len(eigenvectors), classifications*100))
