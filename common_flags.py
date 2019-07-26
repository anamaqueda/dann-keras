import gflags

FLAGS = gflags.FLAGS

# Input data
gflags.DEFINE_integer('img_width', 28, 'Target image width.')
gflags.DEFINE_integer('img_height', 28, 'Target image height.')
gflags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either rgb or grayscale.')

# Training parameters
gflags.DEFINE_integer('batch_size', 32, 'Half of batch size.')
gflags.DEFINE_integer('epochs', 500, 'Number of epochs for training.')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate in epochs for saving model.')

# Files
gflags.DEFINE_string('model_dir', "./model", 'Path to folder containing all the logs, model weights and results.')
gflags.DEFINE_string('train_dir', "../train", 'Path to folder containing training data.')
gflags.DEFINE_string('val_dir', "../val", 'Path to folder containing validation data.')
gflags.DEFINE_string('test_dir', "../test", 'Path to folder containing testing data.')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether restoring a pre-trained model for training.')
gflags.DEFINE_string('weights_fname', None, 'File name of model weights.')
gflags.DEFINE_string('json_model_fname', "model_struct.json", 'File name of model architecture (json serialization).')



