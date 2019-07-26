import numpy as np
import os
import sys
import gflags
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras import backend as K

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


def custom_cce(y_true, y_pred):
    """
    Compute average categorical cross-entropy (CCE) from source samples only.
    """
    # Only first half of the bath contain source samples
    source_true = y_true[:FLAGS.batch_size]
    source_pred = y_pred[:FLAGS.batch_size]
    loss = K.categorical_crossentropy(source_true, source_pred)
    return K.mean(loss)


def custom_bce(y_true, y_pred):
    """
    Compute average binary cross-entropy (BCE).
    """
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return K.mean(loss)


def custom_acc_source(y_true, y_pred):
    """
    Compute average categorical accuracy (CA) from source samples only.
    """
    # Only first half of the bath contain source samples
    source_true = y_true[:FLAGS.batch_size]
    source_pred = y_pred[:FLAGS.batch_size]
    accuracy = K.cast(K.equal(K.argmax(source_true, axis=-1), K.argmax(source_pred, axis=-1)), K.floatx())
    return K.mean(accuracy)


def custom_acc_target(y_true, y_pred):
    """
    Compute average categorical accuracy (CA) from target samples only.
    """
    # Only first half of the bath contain target samples
    target_true = y_true[FLAGS.batch_size:]
    target_pred = y_pred[FLAGS.batch_size:]
    accuracy = K.cast(K.equal(K.argmax(target_true, axis=-1), K.argmax(target_pred, axis=-1)), K.floatx())
    return K.mean(accuracy)


def custom_bin_acc(y_true, y_pred):
    """
    Compute average binary accuracy (BA).
    """
    accuracy = K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    return K.mean(accuracy)


def train_model(train_data, val_data, output_dim, model):
    """
    Model training.
    :param train_data: a DataLoader instance for training data.
    :param val_data: a DataLoader instance for validation data.
    :param output_dim: output dimension of the label predictor (number of classes).
    :param model: a Model instance to be trained.
    """
    # Iterator generating training data batch by batch
    train_generator = data_utils.batch_generator(train_data, output_dim, FLAGS.batch_size, shuffle=True)

    # Iterator generating validation data batch by batch
    val_generator = data_utils.batch_generator(val_data, output_dim, FLAGS.batch_size, shuffle=True)

    # Configure training process
    model.compile(loss=[custom_cce, custom_bce], metrics=[custom_acc_source, custom_acc_target, custom_bin_acc],
                  optimizer=SGD(momentum=0.9))

    # Save model with the lowest validation loss for the label predictor (LP)
    lp_weights_path = os.path.join(FLAGS.model_dir, 'lp_weights_{epoch:03d}.h5')
    best_lp_model = ModelCheckpoint(filepath=lp_weights_path, monitor='val_activation_5_loss',
                                    save_best_only=True, save_weights_only=True)

    # Save model with the highest validation loss for the domain classifier (DC)
    dc_weights_path = os.path.join(FLAGS.model_dir, 'dc_weights_{epoch:03d}.h5')
    worst_dc_model = ModelCheckpoint(filepath=dc_weights_path, monitor='val_activation_7_loss',
                                     save_best_only=True, mode='max', save_weights_only=True)

    # Current step (modified during callback to compute the training progress linearly changing from 0 to 1)
    model.current_step = K.variable(0.0)

    # Number of total training steps
    steps_per_epoch = int(np.ceil((train_data.num_source + train_data.num_target) / (FLAGS.batch_size*2)))
    total_steps = FLAGS.epochs * steps_per_epoch

    # Customized callback
    # - Save logs of training and validation losses
    # - Modify hp_lambda value (not used)
    # - Save model every 'log_rate' epochs
    my_callback = log_utils.MyCallback(FLAGS.model_dir, FLAGS.log_rate, FLAGS.batch_size, total_steps)

    # Train model
    validation_steps = int(np.ceil((val_data.num_source + val_data.num_target) / (FLAGS.batch_size*2)))
    model.fit_generator(train_generator,
                        epochs=FLAGS.epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[best_lp_model, worst_dc_model, my_callback, TensorBoard(log_dir=FLAGS.model_dir)],
                        validation_data=val_generator,
                        validation_steps=validation_steps)

    # Set up loss storage vector
    logs = {'train': my_callback.train_logs, 'val': my_callback.val_logs}
    utils.write_to_file(logs, os.path.join(FLAGS.model_dir, 'logs.json'))


def _main():
    # Set random seed
    #seed = np.random.randint(0, 2 * 31 - 1)
    np.random.seed(5)
    tf.set_random_seed(5)

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

    # Read training data
    train_data = data_utils.DataLoader(FLAGS.train_dir, FLAGS.model_dir, output_dim, FLAGS.img_mode,
                                       target_size=(FLAGS.img_height, FLAGS.img_width), is_train=True)

    # Read validation data
    val_data = data_utils.DataLoader(FLAGS.val_dir, FLAGS.model_dir, output_dim, FLAGS.img_mode,
                                     target_size=(FLAGS.img_height, FLAGS.img_width))
    val_data.pixel_mean = train_data.pixel_mean

    # Initialize model
    model = nets.dann_mnist(FLAGS.img_width, FLAGS.img_height, img_channels, output_dim)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.model_dir, FLAGS.json_model_fname)
    utils.model_to_json(model, json_model_path)

    # Train model
    train_model(train_data, val_data, output_dim, model)
    
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
