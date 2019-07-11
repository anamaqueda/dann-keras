import gflags

FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input data
gflags.DEFINE_integer('img_width', 28, 'Target Image Width')
gflags.DEFINE_integer('img_height', 28, 'Target Image Height')
gflags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either rgb or grayscale')

# Training parameters
gflags.DEFINE_integer('batch_size', 32, 'Half of batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 500, 'Number of epochs for training')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')

# Files
gflags.DEFINE_string('model_dir', "./models", 'Folder containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "../training", 'Folder containing training experiments')
gflags.DEFINE_string('val_dir', "../validation", 'Folder containing validation experiments')
gflags.DEFINE_string('test_dir', "../testing", 'Folder containing testing experiments')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained model for training')
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) filename of model weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json", 'Model struct json serialization, filename')



