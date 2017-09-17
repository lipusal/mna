import argparse
import numpy as np
from mna.tp01.utils.images import *
# import matplotlib.pyplot as plt
from sklearn import svm

parser = argparse.ArgumentParser(description="Face recognizer. Receives an image and identifies it in a database.")
parser.add_argument("directory", help="Directory from which to load images. Directory should have further "
                                      "subdirectories, where each subdirectory belongs to a different individual. Each "
                                      "individual should have the same amount of pictures.", type=str)
parser.add_argument("num_train", help="Number of pictures per individual to take as training pictures.", type=int)
parser.add_argument("num_test", help="Number of pictures per individual to take as testing pictures.", type=int)
parser.add_argument("--verbose", "-v", help="Print verbose information while running", action="store_true",
                    default=False)
parser.add_argument("--cutoff", "-c", help="Percentage of captured variance at which to cut off using eigenvectors. "
                                           "Decimal in (0, 1]. Default is 0.9", type=float, default=0.9)
parser.add_argument("--time", "-t", help="Print elapsed program time", action="store_true", default=False)
args = parser.parse_args()

if args.time:
    import mna.tp01.utils.timer

# Open images and separate them in training and testing groups
if args.verbose:
    print("Loading pictures...")

num_individuals, images = open_images(args)

# Normalize training images
if args.verbose:
    print("Normalizing pictures...")

normalize_images(images)



# Calculate covariance matrix
# cov = np.cov(images, rowvar=True)
# eigenvalues, eigenvectors = np.linalg.eig(cov)

if args.verbose:
    print("Finding eigenvectors...")

# TODO: Use our functions for this
_, singular_values, eigenvectors = np.linalg.svd(images, full_matrices=False)
eigenvalues = singular_values ** 2

# Get enough eigenvalues to capture at least the specified variance
cummulative_sum = 0
eigenvalues_sum = sum(eigenvalues)

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
if args.verbose:
    print("Projecting images to eigenvectors...")

# Separate images into training and testing
train_images, test_images = separate_images(images, args.num_train, args.num_test)

projected_train_imgs = np.dot(train_images, np.transpose(eigenvectors))
projected_test_imgs = np.dot(test_images, np.transpose(eigenvectors))

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

# Get which pictures belong to which individual
train_classes = list()
test_classes = list()
for i in range(num_individuals):
    train_classes += [i] * args.num_train
    test_classes += [i] * args.num_test

if args.verbose:
    print("Training and testing picture categories...")

clf.fit(projected_train_imgs, train_classes)
classifications = clf.score(projected_test_imgs, test_classes)
print('Precisión con {0} autocaras: {1}%'.format(len(eigenvectors), classifications*100))
