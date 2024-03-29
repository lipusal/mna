import argparse

from cv2 import cv2
import mna.tp01.open_cv.FaceDetection as fd

import numpy as np
import time

from mna.tp01.utils.EigAlgorithm import EigAlgorithm
from mna.tp01.utils.Images import *
# import matplotlib.pyplot as plt
import os
from sklearn import svm

from mna.tp01.utils.QRAlgorithm import QRAlgorithm

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
    import mna.util.timer

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

# TODO: El código de referencia no hace esto, ver por qué no centramos las fotos así en vez de con lo de abajo
# # Subtract mean from train images
# mean_face = mean_image(train_images)
# train_images -= mean_face
# # Also center testing images with respect to training images
# test_images -= mean_face

if args.verbose:
    print("Computing kernel matrices...")

# Define and compute testing and training kernels
# Both kernels are polynomial of degree `degree`
degree = 2

file = open('/Users/juanlipuma/Desktop/data.txt', 'w')
file.write("DENOMINATOR\tCONSTANT\tPRECISION\n")

for CONSTANT in range(100+1):
    for DENOMINATOR in range(1, num_train_pics+2):
        train_kernel = ((np.dot(train_images, train_images.T)/DENOMINATOR) + CONSTANT) ** degree
        # Center kernel in the high-dimension space with the following trick (ver filmina 27 de teórica Kernel Trick)
        ones_train = np.ones([num_train_pics, num_train_pics]) / num_train_pics
        train_kernel = train_kernel - np.dot(ones_train, train_kernel) - np.dot(train_kernel, ones_train) + np.dot(ones_train, np.dot(train_kernel, ones_train))

        test_kernel = ((np.dot(test_images, train_images.T)/DENOMINATOR) + CONSTANT) ** degree
        # Also center test kernel
        ones_test = np.ones([num_test_pics, num_train_pics]) / num_train_pics
        test_kernel = test_kernel - np.dot(ones_test, train_kernel) - np.dot(test_kernel, ones_train) + np.dot(ones_test, np.dot(train_kernel, ones_train))

        if args.verbose:
            print("Finding eigenfaces...")

        # TODO: Use our functions for this
        t0 = time.time()
        # eigenvalues, eigenfaces = EigAlgorithm.wilkinsonEig(train_kernel, QRAlgorithm.HouseHolder)
        print("It took " + str(time.time()-t0) + "seconds ")
        # eigenvalues, subEigenFaces = np.linalg.eig(train_images.dot(train_images.T))
        # eigenfaces = train_images.T.dot(subEigenFaces.T).T
        eigenvalues, eigenfaces = np.linalg.eigh(train_kernel)
        # Get the keys that would sort eigenvalues by descending absolute value
        keys = np.argsort(np.absolute(eigenvalues))[::-1]
        # Sort eigenvalues, and their corresponding eigenfaces, by these keys
        eigenvalues = np.absolute([eigenvalues[key] for key in keys])
        eigenfaces = np.asarray([eigenfaces[:, key] for key in keys])

        # Normalize eigenfaces (i.e. divide by their norm)
        for i in range(len(eigenfaces)):
            eigenfaces[:, i] = eigenfaces[:, i] / np.sqrt(eigenvalues[i])

        # Get enough eigenvalues to capture at least the specified variance
        cumulative_sum = 0
        eigenvalues_sum = sum(eigenvalues)

        used_eigenfaces = 0
        for i in range(len(eigenfaces)):
            used_eigenfaces += 1
            cumulative_sum += eigenvalues[i]
            if cumulative_sum/eigenvalues_sum >= args.cutoff:
                break

        if args.verbose:
            if cumulative_sum / eigenvalues_sum < args.cutoff:
                print("[WARN]: Couldn't capture desired variance (%g) with all %i eigenfaces (captured %g), continuing"
                      % (args.cutoff, len(eigenfaces), cumulative_sum / eigenvalues_sum))
            else:
                print("Captured desired variance (%g) with %i/%i eigenfaces" % (args.cutoff, used_eigenfaces, len(eigenfaces)))

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

        # Get which pictures belong to which subdirectory
        subdirs = [os.path.split(subdir)[-1] for subdir in get_subdirs(args.directory)]
        train_classes = list()
        test_classes = list()
        for i in range(num_individuals):
            # Directories were traversed in the same order as `subdirs`, so we know we are putting the right directory to each
            # picture
            train_classes += [subdirs[i]] * args.num_train
            test_classes += [subdirs[i]] * args.num_test

        if args.verbose:
            print("Training and testing picture categories...")

        # Truncate number of used eigenfaces
        projected_train_imgs = projected_train_imgs[:, 0:used_eigenfaces]
        projected_test_imgs = projected_test_imgs[:, 0:used_eigenfaces]

        clf.fit(projected_train_imgs, train_classes)
        classifications = clf.score(projected_test_imgs, test_classes)
        precision = classifications*100
        print("Classification accuracy with %i eigenfaces: %g%%" % (used_eigenfaces, precision))

        print('Writing "%g\t%g\t%g" to file' % (DENOMINATOR, CONSTANT, precision))
        file.write('%g\t%g\t%g\n' % (DENOMINATOR, CONSTANT, precision))
        file.flush()

print("Done")
file.close()
