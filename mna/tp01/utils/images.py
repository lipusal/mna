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

    pics_per_individual = args.num_train + args.num_test

    # Take very first picture size as default size
    first_dir = normalize_dir(subdirs[0], base_dir)
    first_pic = normalize_path(listdir(first_dir)[0], first_dir)
    picture_size = len(to_grayscale(first_pic))

    for individual in subdirs:
        # Expected picture names: 1.pgm, 2.pgm, ..., n.pgm
        pics = [normalize_path("%i.pgm" % index, individual) for index in range(1, pics_per_individual + 1)]
        # Keep only existing pictures
        pics = [pic for pic in pics if path.exists(pic)]

        if len(pics) != pics_per_individual:
            raise Exception("Expecting %i pictures per individual, of the form 1.pgm, 2.pgm, ..., %i.pgm. but found "
                            "%i in %s" % (pics_per_individual, pics_per_individual, len(pics), individual))

        train_pics, test_pics = process_pics(pics, picture_size, args.num_train, args.num_test)
        result[0] += train_pics
        result[1] += test_pics

    return num_subdirs, result[0], result[1]


def mean_image(images):
    return np.mean(images, 0)


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


# TODO: Delete if unused
def separate_images(images, num_train, num_test):
    """Separate the given images (assuming they are contiguous by individual) into training images and testing
    images."""
    result = [[], []]

    index = 0
    while index < len(images):
        # Add training
        for i in range(num_train):
            result[0].append(images[index])
            index += 1
        # Add testing
        for i in range(num_test):
            result[1].append(images[index])
            index += 1

    return result
