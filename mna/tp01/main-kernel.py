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
parser.add_argument("--cutoff", "-c", help="Percentage of captured variance at which to cut off using eigenfaces. "
                                           "Decimal in (0, 1]. Default is 0.9", type=float, default=0.9)
parser.add_argument("--time", "-t", help="Print elapsed program time", action="store_true", default=False)
args = parser.parse_args()

if args.time:
    import mna.tp01.utils.timer

# Open images and separate them in training and testing groups
if args.verbose:
    print("Loading pictures...")

num_individuals, train_images, test_images = open_images(args)
num_train_pics = num_individuals * args.num_train
num_test_pics = num_individuals * args.num_test


# Normalize training images
if args.verbose:
    print("Normalizing pictures...")

# Make numbers easier to play with
# TODO: Define whether ** 2 stays or not
train_images = (np.asarray(train_images) - 127.5) / 127.5
test_images = (np.asarray(test_images) - 127.5) / 127.5

# TODO: Centrar fotos con la manera flashera
# Subtract mean from train images
mean_face = mean_image(train_images)
train_images -= mean_face
# Also center testing images with respect to training images
test_images -= mean_face

if args.verbose:
    print("Finding kernel matrix...")

# Define and compute testing and training kernels
# Both kernels are polynomial of degree `degree`
degree = 2

train_kernel = ((np.dot(train_images, train_images.T)/num_train_pics) + 1) ** degree
# Center kernel (????)
ones_m = np.ones([num_train_pics, num_train_pics]) / num_train_pics
train_kernel = train_kernel - np.dot(ones_m, train_kernel) - np.dot(train_kernel, ones_m) + np.dot(ones_m, np.dot(train_kernel, ones_m))

test_kernel = ((np.dot(test_images, train_images.T)/num_train_pics) + 1) ** degree
# Center test kernel (????)
ones_ml = np.ones([num_test_pics, num_train_pics]) / num_train_pics
test_kernel = test_kernel - np.dot(ones_ml, train_kernel) - np.dot(test_kernel, ones_m) + np.dot(ones_ml, np.dot(train_kernel, ones_m))


if args.verbose:
    print("Finding eigenfaces...")

# TODO: Use our functions for this
eigenvalues, eigenfaces = np.linalg.eigh(train_kernel)
# Eigenvalues/vectors are returned in ascending order, flip to descending
eigenvalues = np.flipud(eigenvalues)
eigenfaces = np.fliplr(eigenfaces)
# Normalize eigenfaces (i.e. divide by their norm)
for i in range(len(eigenfaces)):
    if i == 359:
        2-2
    eigenfaces[:, i] = eigenfaces[:, i] / np.sqrt(eigenvalues[i])

# Get enough eigenvalues to capture at least the specified variance
cummulative_sum = 0
eigenvalues_sum = sum(eigenvalues)

used_eigenfaces = 0
for i in range(len(eigenfaces)):
    used_eigenfaces += 1
    cummulative_sum += eigenvalues[i]
    if cummulative_sum/eigenvalues_sum >= args.cutoff:
        break

if args.verbose:
    if cummulative_sum / eigenvalues_sum < args.cutoff:
        print("[WARN]: Couldn't capture desired variance (%g) with all %i eigenfaces, continuing"
              % (args.cutoff, len(eigenfaces)))
    else:
        print("Captured desired variance (%g) with %i/%i eigenfaces" % (args.cutoff, used_eigenfaces, len(eigenfaces)))

# eigenfaces = eigenfaces[0:used_eigenfaces]

# Project each image to all chosen eigenfaces
if args.verbose:
    print("Projecting images to eigenfaces...")

projected_train_imgs = np.dot(train_kernel.T, eigenfaces)
projected_test_imgs = np.dot(test_kernel, eigenfaces)

# Show mean face
# fig, axes = plt.subplots(1,1)
# axes.imshow(np.reshape(mean_pixels,[112,92]),cmap='gray')
# fig.suptitle('Mean face')
# fig.show()

# Show all eigenfaces
# for eigenface in eigenfaces:
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

clf.fit(projected_train_imgs[0:used_eigenfaces], train_classes)
classifications = clf.score(projected_test_imgs[0:used_eigenfaces], test_classes)
print("Classification accuracy with %i eigenfaces: %g%%" % (used_eigenfaces, classifications*100))
