import argparse
import numpy as np
from mna.tp01.utils.Images import *
from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.utils.EigAlgorithm import *
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import os
from sklearn import svm
import time
import cv2
import mna.tp01.open_cv.FaceDetection as fd

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
    import mna.tp01.utils.Timer

# Open images and separate them in training and testing groups
if args.verbose:
    print("Loading pictures...")

num_individuals, train_images, test_images = open_images(args)

# Normalize training images
if args.verbose:
    print("Normalizing pictures...")

# Make numbers easier to play with
# TODO: Define whether ** 2 stays or not
train_images = (np.asarray(train_images) / 255.0)
test_images = (np.asarray(test_images) / 255.0)

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
eigenvalues, subEigenFaces = EigAlgorithm.wilkinsonEig(train_images.dot(train_images.T), QRAlgorithm.HouseHolder)
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

eigen2 = (np.reshape(eigenfaces[1,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Segunda autocara')

eigen3 = (np.reshape(eigenfaces[2,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Tercera autocara')
print(test_classes)

clf.fit(projected_train_imgs, train_classes)
classifications = clf.score(projected_test_imgs, test_classes)
print("Classification accuracy with %i eigenfaces: %g%%" % (used_eigenfaces, classifications*100))

cascPath = "./open_cv/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

print(os.path.dirname(os.path.realpath(__file__)))
i=0
while True:

    framesTaken = 0
    success = 0
    username = input("Enter your username: ")
    video_capture = cv2.VideoCapture(0)
    while framesTaken <= 20:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            # print("The video capture is not working.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(92, 112)
        )

        # Draw a rectangle around the faces
        for f in faces:
            x, y, w, h = fd.resizeFace(f)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            newImg = fd.cropImage(frame, fd.resizeFace(f))
            newImg = fd.resizeImg(newImg)
            newImg = newImg.convert('L')
            newImg = np.array(newImg).ravel()
            newImg = (np.array(newImg) / 255.0) - mean_face
            newImg = np.dot(np.array(newImg), eigenfaces.T)
            name = clf.predict([newImg])
            framesTaken += 1
            if name[0] == username:
                success += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord(' ') and frame is not None and len(faces) > 0:
            newImg = fd.cropImage(frame, fd.resizeFace(faces[0]))
            newImg = fd.resizeImg(newImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    print(success)

    if success >= 5:
        print("Welcome " + username + ", type 'exit' to exit")
        while True:
            exit = input()
            if exit == "exit":
                break
    else:
        print("You are not " + username + " , don't try to fool me.")
