import numpy as np
import os
import sys
import gflags
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras import backend as K

import logz
import nets
import utils
import data_utils
import log_utils
from common_flags import FLAGS


# Set GPU 1 for training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Constants
TRAIN_PHASE = 1


def get_model(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model
    :param img_width: target image width.
    :param img_height: target image height.
    :param img_channels: target image channels.
    :param output_dim: dimension of model output.
    :param weights_path: path to pre-trained weights.
    :return: a Model instance.
    """
    model = nets.dann_mnist(img_width, img_height, img_channels, output_dim)

    # Load pre-trained weights if provided
    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            raise ValueError("Impossible to find weight path. Returning untrained model")

    return model


def custom_categorical_crossentropy(y_true, y_pred):
    """
    Compute average categorical cross-entropy from source samples only.
    :param y_true:
    :param y_pred:
    :return:
    """
    # Only first half of the bath contain source samples
    source_true = y_true[:FLAGS.batch_size]
    source_pred = y_pred[:FLAGS.batch_size]
    loss = K.categorical_crossentropy(source_true, source_pred)
    return K.mean(loss)


def custom_binary_crossentropy(y_true, y_pred):
    """
    Compute average binary cross-entropy.
    :param y_true:
    :param y_pred:
    :return:
    """
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return K.mean(loss)


def train_model(train_data, val_data, output_dim, model, initial_epoch):
    """
    Model training.
    :param train_data: a DataLoader instance for training data.
    :param val_data: a DataLoader instance for validation data.
    :param output_dim:
    :param model: a Model instance to be trained.
    :param initial_epoch: epoch from which training starts.
    """
    # Iterator generating training data batch by batch
    train_generator = data_utils.batch_generator(train_data, output_dim, FLAGS.batch_size, shuffle=True)

    # Iterator generating validation data batch by batch
    val_generator = data_utils.batch_generator(val_data, output_dim, FLAGS.batch_size, shuffle=True)

    # Configure training process
    model.compile(loss=[custom_categorical_crossentropy, custom_binary_crossentropy], optimizer=SGD(momentum=0.9))

    # Save model with the lowest validation loss for the label predictor (LP)
    lp_weights_path = os.path.join(FLAGS.model_dir, 'lp_weights_{epoch:03d}.h5')
    best_lp_model = ModelCheckpoint(filepath=lp_weights_path, monitor='val_activation_5_loss',
                                    save_best_only=True, save_weights_only=True)

    # Save model with the highest validation loss for the domain classifier (DC)
    dc_weights_path = os.path.join(FLAGS.model_dir, 'dc_weights_{epoch:03d}.h5')
    worst_dc_model = ModelCheckpoint(filepath=dc_weights_path, monitor='val_activation_7_loss',
                                     save_best_only=True, mode='max', save_weights_only=True)

    # Number of total steps
    total_steps = FLAGS.epochs * (train_data.num_source + train_data.num_target) / (FLAGS.batch_size * 2)

    # Customized callback
    # - Save logs of training and validation losses.
    # - Modify hp_lambda value
    logz.configure_output_dir(FLAGS.model_dir)
    save_model_and_loss = log_utils.MyCallback(FLAGS.model_dir, FLAGS.log_rate, FLAGS.batch_size, total_steps)

    # Train model
    steps_per_epoch = int(np.ceil((train_data.num_source + train_data.num_target) / (FLAGS.batch_size*2)))
    validation_steps = int(np.ceil((val_data.num_source + val_data.num_target) / (FLAGS.batch_size*2)))
    model.fit_generator(train_generator,
                        epochs=FLAGS.epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[best_lp_model, worst_dc_model, save_model_and_loss, TensorBoard(log_dir=FLAGS.model_dir)],
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)


def _main():
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0, 2 * 31 - 1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Set training phase
    K.set_learning_phase(TRAIN_PHASE)

    # Create the experiment root directory if not already there
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    
    # Image mode (RGB or grayscale)
    if FLAGS.img_mode == 'rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    # Output dimension (10 classes: digits from 0 to 9)
    output_dim = 10

    # Generate training data
    train_data = data_utils.DataLoader(FLAGS.train_dir, FLAGS.model_dir, output_dim, FLAGS.img_mode,
                                       target_size=(FLAGS.img_height, FLAGS.img_width), is_train=True)

    # Generate validation data
    val_data = data_utils.DataLoader(FLAGS.val_dir, FLAGS.model_dir, output_dim, FLAGS.img_mode,
                                     target_size=(FLAGS.img_height, FLAGS.img_width))
    val_data.pixel_mean = train_data.pixel_mean

    # Weights to restore
    weights_path = os.path.join(FLAGS.model_dir, FLAGS.weights_fname)
    
    # Epoch from which training starts
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights are initialized randomly
        weights_path = None
    else:
        # In this case weights are initialized as specified in pre-trained model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = get_model(FLAGS.img_width, FLAGS.img_height, img_channels, output_dim, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.model_dir, FLAGS.json_model_fname)
    utils.model_to_json(model, json_model_path)

    # Train model
    train_model(train_data, val_data, output_dim, model, initial_epoch)
    
    # Plot training and validation losses
    utils.plot_loss(FLAGS.model_dir)


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
