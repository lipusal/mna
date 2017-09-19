import argparse
import numpy as np
from mna.tp01.utils.images import *
from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.utils.HouseHolder import *
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import os
from sklearn import svm
import time

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

# Normalize training images
if args.verbose:
    print("Normalizing pictures...")

# Make numbers easier to play with
# TODO: Define whether ** 2 stays or not
train_images = (np.asarray(train_images) / 255.0) ** 2
test_images = (np.asarray(test_images) / 255.0) ** 2

# Subtract mean from train images
mean_face = mean_image(train_images)
train_images -= mean_face
# Also center testing images with respect to training images
test_images -= mean_face



# Calculate covariance matrix
# cov = np.cov(images, rowvar=True)
# eigenvalues, eigenfaces = np.linalg.eig(cov)

if args.verbose:
    print("Finding eigenfaces...")

# TODO: Use our functions for this
# _, singular_values, eigenfaces2 = np.linalg.svd(train_images, full_matrices=False)
# eigenvalues = singular_values ** 2
t0 = time.time()
eigenvalues, subEigenFaces = QRAlgorithm.wilkinsonEig(train_images.dot(train_images.T), HouseHolder.qr)
# eigenvalues, subEigenFaces = QRAlgorithm.wilkinsonEig(QRAlgorithm.HessenbergReduction(train_images.dot(train_images.T)), HouseHolder.qr)
print("It took " + str(time.time()-t0) + "seconds ")
# eigenvalues, subEigenFaces = np.linalg.eig(train_images.dot(train_images.T))
eigenfaces = train_images.T.dot(subEigenFaces.T).T

row_sums = np.linalg.norm(eigenfaces, axis=1)
eigenfaces = np.divide(eigenfaces,col(row_sums))
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

eigenfaces = eigenfaces[0:used_eigenfaces]

# Project each image to all chosen eigenfaces
if args.verbose:
    print("Projecting images to eigenfaces...")

projected_train_imgs = np.dot(train_images, eigenfaces.T)
projected_test_imgs = np.dot(test_images, eigenfaces.T)

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

    #Primera autocara...
    horsize     = 92
versize     = 112
eigen1 = (np.reshape(eigenfaces[0,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen1,cmap='gray')
fig.suptitle('Primera autocara')
fig.show()

eigen2 = (np.reshape(eigenfaces[1,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Segunda autocara')
fig.show()

eigen3 = (np.reshape(eigenfaces[2,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Tercera autocara')
fig.show()

clf.fit(projected_train_imgs, train_classes)
classifications = clf.score(projected_test_imgs, test_classes)
print("Classification accuracy with %i eigenfaces: %g%%" % (used_eigenfaces, classifications*100))
