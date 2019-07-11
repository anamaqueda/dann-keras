import os
import numpy as np
import cv2
import json

from keras import backend as K
from keras.utils import to_categorical

import utils


class DataLoader:
    """
    Class for managing data loading of images and labels from both source and target domains.

    We assume that the folder structure is:
    root_folder/
           source/
                class_1/
                    crop_00000000.png
                    crop_00000001.png
                    .
                    .
                    crop_00999999.png
                class_2/
                .
                .
                class_n
           target/
                class_1/
                    crop_00000000.png
                    crop_00000001.png
                    .
                    .
                    crop_00999999.png
                class_2/
                .
                .
                class_n

    :param data_dir: path to the root directory to read data from (train, val, test).
    :param model_dir: path where model weights and logs are saved.
    :param output_dim: output dimension of the label predictor.
    :param img_mode: one of `"rgb"`, `"grayscale"`. Color model to read images.
    :param is_train: whether training data or other.
    :param target_size: tuple of integers. Dimensions to resize input images to.
    """

    def __init__(self, data_dir, model_dir, output_dim, img_mode, target_size=(224, 224), is_train=False):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dim = output_dim
        self.target_size = target_size

        # Initialize image mode
        if img_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', img_mode, '; expected "rgb" or "grayscale".')
        self.img_mode = img_mode

        # Valid image formats
        self.formats = {'png', 'jpg'}

        # File names of all samples/images under 'data_dir'.
        self.source_filenames = []
        self.target_filenames = []

        # Category and domain labels (ground truth) of all samples/images under 'data_dir'.
        self.source_labels = []
        self.target_labels = []

        # Count number of source and target samples
        self.num_source = 0
        self.num_target = 0

        # Decode dataset structure
        domains = []
        categories = []
        for domain in sorted(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, domain)):
                domains.append(domain)
                for category in sorted(os.listdir(os.path.join(data_dir, domain))):
                    if os.path.isdir(os.path.join(data_dir, domain, category)):
                        categories.append(category)
                # Check if number of classes is the same as specified in 'output_dim' for every domain
                assert len(categories) == self.output_dim, \
                    'Dimension of LP output is {} but {} classes were found.'.format(self.output_dim, len(categories))
                if domain == 'source':
                    categories = []

        # Check if pixel mean is already computed
        self.pixel_mean = 0
        compute_mean = False
        if not os.path.isfile(os.path.join(model_dir, 'mean.json')):
            if is_train:
                compute_mean = True
            else:
                raise IOError('mean.json file not found.')
        else:
            with open(os.path.join(model_dir, 'mean.json')) as f:
                mean_dict = json.load(f)
                self.pixel_mean = np.array(mean_dict['pixel_mean'])

        # Read images and ground truth. File names and ground truth are loaded in memory from the beginning. Images
        # instead, will be loaded iteratively as the same time the training process needs a new batch by a generator.
        self.source_images = []
        self.target_images = []
        for dom_id, domain in enumerate(domains):
            for cat_id, category in enumerate(categories):
                cat_path = os.path.join(data_dir, domain, category)
                if os.path.isdir(cat_path):
                    self._decode_experiment_dir(cat_path, domain == 'source', cat_id, compute_mean)

        # Check if some source or target dataset is empty
        if self.num_source == 0:
            raise IOError("Did not find any data on source directory")
        if self.num_target == 0:
            raise IOError("Did not find any data on target directory")

        # Compute mean if necessary
        if compute_mean:
            self._compute_mean()
        self.source_images = None
        self.target_images = None

        # Conversion of list into array
        self.source_labels = np.array(self.source_labels, dtype=K.floatx())
        self.target_labels = np.array(self.target_labels, dtype=K.floatx())

        print('Found {} source images and {} target images belonging to {} classes.'.format(self.num_source,
                                                                                            self.num_target,
                                                                                            self.output_dim))

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath), key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, cat_path, category_id, source, compute_mean):
        """
        Extract valid filenames and corresponding ground truth from every class/category.
        :param cat_path: path to class folder to be decoded.
        :param category_id: int, class identifier.
        :param source: boolean, whether domain is source.
        :param is_train: boolean, whether data is for training.
        """
        for root, _, files in self._recursive_list(cat_path):
            for frame_number, fname in enumerate(files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    image_path = os.path.join(root, fname)
                    if source:
                        self.num_source += 1
                        self.source_filenames.append(image_path)
                        self.source_labels.append(category_id)
                        if compute_mean:
                            self.source_images.append(load_img(image_path,
                                                               self.img_mode == 'grayscale',
                                                               self.target_size))
                    else:
                        self.num_target += 1
                        self.target_filenames.append(image_path)
                        self.target_labels.append(category_id)
                        if compute_mean:
                            self.target_images.append(load_img(image_path,
                                                               self.img_mode == 'grayscale',
                                                               self.target_size))

    def _compute_mean(self):
        # Stack source and target images
        self.source_images = np.array(self.source_images)
        self.target_images = np.array(self.target_images)
        all_images = np.vstack([self.source_images, self.target_images])

        # Compute mean value
        self.pixel_mean = all_images.mean((0, 1, 2))

        # Save pixel mean in a dictionary
        mean_dict = {'pixel_mean': self.pixel_mean.tolist()}
        utils.write_to_file(mean_dict, os.path.join(self.model_dir, 'mean.json'))


def batch_generator(data, output_dim, batch_size, shuffle=True):
    """
    Generate batches of data.
    Given a list of numpy data, it iterates over the list and returns batches of the same size.
    :param data: a DataLoader instance.
    :param output_dim:
    :param batch_size: size of each domain
    :param shuffle: Whether shuffle data.
    """
    if data.img_mode == 'rgb':
        image_shape = data.target_size + (3,)
    else:
        image_shape = data.target_size + (1,)

    source_indices = range(data.num_source)
    target_indices = range(data.num_target)

    # Shuffle data
    if shuffle:
        source_indices = np.random.permutation(data.num_source)
        target_indices = np.random.permutation(data.num_target)

    batch_count = 0
    while True:
        # If all source or target data has been seen, start batch count
        if batch_count * batch_size + batch_size >= data.num_target or \
                batch_count * batch_size + batch_size >= data.num_source:
            batch_count = 0
            # Shuffle data
            if shuffle:
                source_indices = np.random.permutation(data.num_source)
                target_indices = np.random.permutation(data.num_target)

        # Define start and end of mini-batch
        start = batch_count * batch_size
        end = start + batch_size

        # Create mini-batch of source and target indices
        mini_batch_indices_source = source_indices[start:end]
        mini_batch_indices_target = target_indices[start:end]
        batch_count += 1

        # Create mini-batch of source images
        batch_x_s = np.zeros((mini_batch_indices_source.shape[0],) + image_shape, dtype=K.floatx())
        for i, j in enumerate(mini_batch_indices_source):
            fname = data.source_filenames[j]
            batch_x_s[i] = normalize_image(load_img(fname, grayscale=data.img_mode == 'grayscale',
                                                    target_size=data.target_size), data.pixel_mean)
        batch_cat_s = np.array(to_categorical(data.source_labels[mini_batch_indices_source], num_classes=output_dim,
                                              dtype=K.floatx()))
        batch_dom_s = np.zeros((mini_batch_indices_source.shape[0],), dtype=K.floatx())

        # Create mini-batch of target images
        batch_x_t = np.zeros((mini_batch_indices_target.shape[0],) + image_shape, dtype=K.floatx())
        for k, l in enumerate(mini_batch_indices_target):
            fname = data.target_filenames[l]
            batch_x_t[k] = normalize_image(load_img(fname, grayscale=data.img_mode == 'grayscale',
                                                    target_size=data.target_size), data.pixel_mean)
        batch_cat_t = np.array(to_categorical(data.target_labels[mini_batch_indices_target], num_classes=output_dim,
                                              dtype=K.floatx()))
        batch_dom_t = np.ones((mini_batch_indices_target.shape[0],), dtype=K.floatx())

        # Concatenate source (first) and target (second) mini-batches
        batch_x = np.concatenate((batch_x_s, batch_x_t), axis=0)
        batch_cat = np.concatenate((batch_cat_s, batch_cat_t), axis=0)
        batch_dom = np.concatenate((batch_dom_s, batch_dom_t), axis=0)

        # Define model's output
        batch_y = [batch_cat, batch_dom]

        yield batch_x, batch_y


def load_img(path, grayscale=False, target_size=None):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size) or tuple of ints `(img_height, img_width)`.

    # Returns
        Image as numpy array.
    """
    # Read input image
    img = cv2.imread(path)

    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))

    if grayscale:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    img = np.asarray(img, dtype=np.float32)

    return img


def normalize_image(img, pixel_mean):
    """
    Normalize current image by subtracting the mean.
    :param img: image to be normalized.
    :param pixel_mean: mean pixel value.
    :return: normalized image as numpy array
    """
    return (img - pixel_mean) * 1./255.

