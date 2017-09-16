from PIL import Image
from os import path, listdir
import numpy as np


def open_images(args):
    """Take arguments received from command line and process pictures. Traverse subdirectories from the given base
    directory and convert found images, ensuring each individual has the same number of pictures and that all pictures
    have the same size. Return a tuple of the form ([], []), where the first list is composed of the training images and
    the second list is composed of the test images."""
    base_dir = normalize_dir(args.directory)
    # Get all normalized subdirectories, relative to the base directory
    subdirs = [normalize_dir(dir, base_dir) for dir in listdir(base_dir) if path.isdir(path.join(base_dir, dir))]
    num_subdirs = len(subdirs)
    if args.verbose:
        print("Detected %i subdirectories, assuming one per individual" % num_subdirs)

    result = [[], []]

    num_train_per_person = args.num_train
    num_test_per_person = args.num_test
    # Take very first picture size as default size
    first_dir = normalize_dir(subdirs[0], base_dir)
    first_pic = normalize_path(listdir(first_dir)[0], first_dir)
    picture_size = len(to_grayscale(first_pic))

    for individual in subdirs:
        pics = [normalize_path(pic, individual) for pic in listdir(individual)]

        if len(pics) != num_train_per_person + num_test_per_person:
            raise Exception("Expecting %i pictures per individual but found %i in %s" %
                            (num_train_per_person + num_test_per_person, len(pics), individual))

        train_pics, test_pics = process_pics(pics, picture_size, num_train_per_person, num_test_per_person)
        result[0] += train_pics
        result[1] += test_pics

    return num_subdirs, result[0], result[1]


def mean_image(images):
    return np.mean(images, 0)


def normalize_images(images):
    """Subtract mean (per pixel) to all images. Ensure all elements are converted to float."""
    mean_pixels = mean_image(images)
    for i in range(len(images)):
        for j in range(len(images[i])):
            images[i][j] = float(images[i][j]) - mean_pixels[j]

    return images


def normalize_path(raw_path, base_dir=""):
    """Normalize the given raw_path (i.e. make it system-independent), relative to the given base_dir (optional) and
    verify it exists. Raise exception if invalid."""
    normalized_path = path.normpath(path.join(base_dir, raw_path))
    if not path.exists(normalized_path):
        raise Exception("%s does not exist" % raw_path)

    return normalized_path


def normalize_dir(raw_path, base_dir=""):
    """Same as normalize_path but also ensures path is a directory"""
    normalized_dir = normalize_path(raw_path, base_dir)
    if not path.isdir(normalized_dir):
        raise Exception("%s is not a directory" % path)

    return normalized_dir


def process_pics(pics, picture_size, num_train, num_test):
    """Process the given list of pictures, returning a tuple of the form ([num_train], [num_test]) where each element of
    the array is a processed picture."""
    result = ([], [])
    i = 0
    for train_pic in range(num_train):
        converted_pic = to_grayscale(pics[i])

        if len(converted_pic) != picture_size:
            raise Exception("Expecting all images to be of the same size but %s is of a different size" % train_pic)

        result[0].append(converted_pic)
        i += 1
    # FIXME: Avoid duplicated code
    for test_pic in range(num_test):
        converted_pic = to_grayscale(pics[i])

        if len(converted_pic) != picture_size:
            raise Exception("Expecting all images to be of the same size but %s is of a different size" % test_pic)

        result[1].append(converted_pic)
        i += 1

    return result


def to_grayscale(picture_path):
    """Open a picture and convert it to a 1xN array of grayscale numbers (each number is inside [0, 255]), where N is
    the number of pixels in the picture."""
    return list(
        Image.open(picture_path)
             .convert('L')
             .getdata())
