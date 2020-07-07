"""
Contains different functions to load and transform the dataset to prepare for model
training. Also import a couple of functions from model_transform file.
"""

import torch
import random
import torchvision
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms.functional as FT
import xml.etree.ElementTree as ET
# functions import
from model_transform.py import find_jaccard_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

label_map = {k: v+1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_color_map = {v: k for k, v in label_map.items()}

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}   # assigns the colors to classes


# Functions

def parse_annotation(annotation_path):

    """ Transforms the .xml file passed into annotation_path and creates list of
     boxes, labels, and difficulties

     :param annotation_path: path to the .xml file
     :return: lists of objects, labels, difficulties from the .xml file
     """

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()

        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, bounding boxes, labels of the objects in these images,
    and save them to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved

    :return: no explicit return but saves the json files in the output folder param.
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0


    # load the training data

    for path in [voc07_path, voc12_path]:

        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            objects = parse_annotation(os.path.join(path, 'Annotations', id +'.xml'))   # Just transforms the read in xml file a little bit
            if len(objects) == 0:
                continue

            n_objects += len(objects)
            train_objects.append(objects)  # is this creating an appended list or dictionary?
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))
         # train objects contains the dictionary of all the objects in a single image
        #
    assert len(train_objects) == len(train_images)

    # save them to a JSON file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)

    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    with open(os.path.join(output_folder, 'label_map.json'), 'w')as j:
        json.dump(label_map, j)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Test data

    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) ==len(test_images)

    print('\n There are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))

# main transform function that gets called on the dataset

def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformatons on the images

    :param image: Apply transformations on PIL image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions
    :param labels: labels of objects, a tensor of dimensions
    :param difficulties: difficulty of detection, we might not even need this but need to perform transformations on it
    :param split: Split the train and test datasets, since we perform different transformations on each
    :return: return the transformed image,  boxes, labels, difficulties and splitted datasets

    """
    assert split in ('TRAIN', 'TEST')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    if split == 'TRAIN':

        # photometric distortion is not built-in so have to pass that in too

        new_image = photometric_distort(new_image)

        new_image = FT.to_tensor(new_image)

        # expand i.e. zoom out the image with a 50% chance (i.e. applicable to 50% data?)
        # expand with the mean of the whole dataset
        # expand is not in-built, so have to write that function too
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filter=mean)

        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                        new_difficulties)

        new_image = FT.to_pil_image(new_image)

        # flip image with a 50% chance
        # write the flip image function too
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # resize the image (300, 300) - this also converts absolute boundary coordinates to their fractional form
    # squeezenet takes in 224, 224 size images
    new_image, new_boxes = resize(new_image, new_boxes, dims=(224, 224))

    new_image = FT.to_tensor(new_image)

    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties

# supplemental functions

def flip(image, boxes):
    """
    Flip images horizontally.

    :param images: PIL image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimension(n_objects, 4)
    :return: flipped image ,updated bounding box coordinates
    """
    new_image = FT.hflip(image)

    # flip the boxes too:

    new_boxes = boxes

    new_boxes[:, 0] = image.width - boxes[:, 0] - 1  # takes the first values from tensor of alllll the boxes and subtract from image.width
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1  # takes the 3rd value at index 2 from all the tensors
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes


def expand(image, boxes, filler):
    """
    Performs zooming out operation by placing the image in larger canvas of filler material.

    Helps to learn  to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions(n_objects, 4)
    :param filler: RGB values of the filler material, a list like [R, G, B]

    """

    # calculate the dimension of proposed
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_h)

    # filler is passed in with a list of values of R, G, B channels
    filler = torch.FloatTensor(filler)   # 3
    # new we create the filler image on which we are going to impose our own image:
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)
    # outputs: (3, new_h, new_W)
    # after creating a new image with a bunch of zeros, we will super impose the original image
    # to expand it
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # adjust the bounding boxes as well
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes


def resize(image, boxes, dim=(224, 224), return_percent_coords=True):
    """
    Resize Image. SqueezeNet expects them to be at least 224.

    Since we have percent/fractional coordinates for  the bounding boxes (w.r.t the image dimensions),
    we can choose to keep those.

    :param image: image, a PIL image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions(n_objects, 4)
    :return: returns the resize image along with the bounding boxes updates (or fractional coordinates,
    in which case they remain the same because they can be calculated again using the newest dimensions of
    the image)

    """
    # resize using the functional module
    new_image = FT.resize(image, dims)

    # resizing the bounding boxes is the complicated part
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    # we divide the boxes coordinates by the old dimensions of lets say 300, 400, 300, 400 to get
    # the percent dimensions
    new_boxes = boxes / old_dims # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort the saturation, hue, contrast so that the model has a range of situations to train on.
    Each distortion is done with 50% chance

    :param image: image, a PIL image
    :return: distorted image
    """

    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # caffe uses a hue of 18 but we will normalize it since Pytorch needs normalized value:
                adjust_factor = random.uniform(-18 / 255., 18/255.)

            else:
                # lower and upper values for brightness, contrast, and saturation
                adjust_fator = random.uniform(0.5, 1.5)


            new_image = d(new_image, adjust_factor)
            # would do something like:
            # FT.adjust_saturation(image, random number from adjust_factor)
    return new_image



def random_crop(image, boxes, labels, difficulties):
    """
    Randomly crop the image. Helps in learning to detect larger and partial objects

    Note that some objects may be cut out entirely.

    :param image:, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of the objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding boxes corrodinates, updates labels, updated difficulties

    """
    original_h = image.size(1)   # takes the second element in the tensor
    original_w = image.size(1)

    # keep chossing a minimum overlap until a successful crop is made
    while True:
        # randomly draw the value for minimum overlap:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])   # 'None means no cropping

        if min_overlap is None:
            return image, boxes, labels, difficulties

        max_trials = 50

        for _ in range(max_trials):
            # can't go smaller than 0.3 of the image for the crop
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue
            # FIXME: I believe with continue here, it stops going forward and goes back through the
            # loop until the aspect ratio is between 0. and 2

            left = random.randint(0, original_w - new_w)   # 4.2 in the example
            right = left + new_w

            top = random.randint(0, original_h - new_h)     # 3.3
            bottom = top + new_h
            # makes sense since we are still on the grid so we would go to left corner at 4,0
            # for the new crop for the top left corner. Then, we would go to the right corner which would
            # be the left corner coordinate + the new_w which was 6.3 in our case
            # For the bottom, we would do the same with going to the top number and adding new_h to it
            # Finally we performed the crop
            crop = torch.FloatTensor([left, top, right, bottom])

            # find the jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)  (1, n_objects)

            overlap = overlap.squeeze(0)  # (n_objects)

            if overlap.max().item() < min_overlap:
                continue

            # finally crop the image:

            new_image = image[:, top:bottom, left:right]   # (3, new_h, new_w)

            # find the center of the original bounding boxes
            bb_center = (boxes[:, :2] + boxes[:, 2:]) /2.  # (n_objects, 2)

            # find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                                bb_centers[:, 1] < bottom)

            # continue doing so until we do find a crop that has the images with objects centered in the crop as well
            if not centers_in_crop.any():
                continue


            # discard the ones that are not in the new cropped image but if they are:

            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])   # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.max(new_boxes[:, 2:], crop[2:])    # crop[2:] is (right, bottom)
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties
