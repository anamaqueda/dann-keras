import gflags
import numpy as np
import os
import sys
from sklearn import metrics

from keras import backend as K

import utils
import data_utils
from common_flags import FLAGS
from nets import GradientReversal


# Set GPU 1 for training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Constants
TEST_PHASE = 0
LP_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DC_CLASSES = ['source-real', 'target-syn']


def _main():

    # Set testing mode (dropout/batch normalization)
    K.set_learning_phase(TEST_PHASE)
    
    # Output dimension (10 classes: digits from 0 to 9)
    output_dim = 10

    # Generate testing data
    test_data = data_utils.DataLoader(FLAGS.test_dir, FLAGS.model_dir, output_dim, FLAGS.img_mode,
                                      target_size=(FLAGS.img_height, FLAGS.img_width))
    
    # Iterator generating testing data batch by batch
    test_generator = data_utils.batch_generator(test_data, output_dim, FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.model_dir, FLAGS.json_model_fname)
    model = utils.json_to_model(json_model_path, {'GradientReversal': GradientReversal})

    # Load weights
    weights_load_path = os.path.join(FLAGS.model_dir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        raise ValueError("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())

    # Compute predictions
    n_samples = test_data.num_source + test_data.num_target
    nb_batches = int(np.ceil(n_samples / (FLAGS.batch_size*2)))
    pred_lp, pred_dc, categorical_true_lp, true_dc = utils.compute_pred_and_gt(model, test_generator,
                                                                               nb_batches, verbose=1)

    # Convert ground truth of labels and domains to an appropriate format
    true_labels = np.argmax(categorical_true_lp, axis=-1)

    # Convert predictions of labels and domains to an appropriate format
    pred_labels = np.argmax(pred_lp, axis=-1)
    pred_domains = pred_dc.T[0]
    pred_bin_domains = np.zeros_like(pred_domains)
    pred_bin_domains[pred_domains >= 0.5] = 1

    # Visualize domain confusion matrix
    utils.plot_confusion_matrix(true_dc, pred_bin_domains, DC_CLASSES, FLAGS.model_dir, normalize=True,
                                name="confusion_dc.png")
    print('Domain accuracy =', metrics.accuracy_score(true_dc, pred_bin_domains))

    # Visualize label confusion matrix for source only
    source_pred_labels = pred_labels[true_dc == 0]
    source_true_labels = true_labels[true_dc == 0]
    utils.plot_confusion_matrix(source_true_labels, source_pred_labels, LP_CLASSES, FLAGS.model_dir,
                                normalize=True, name="confusion_source_lp.png")
    print('Label accuracy on source data =', metrics.accuracy_score(source_true_labels, source_pred_labels))

    # Visualize label confusion matrix for target only
    target_pred_labels = pred_labels[true_dc == 1]
    target_true_labels = true_labels[true_dc == 1]
    utils.plot_confusion_matrix(target_true_labels, target_pred_labels, LP_CLASSES, FLAGS.model_dir,
                                normalize=True, name="confusion_target_lp.png")
    print('Label accuracy on target data =', metrics.accuracy_score(target_true_labels, target_pred_labels))

    # Visualize precision-recall curve for source only
    source_pred_cat_labels = pred_lp[true_dc == 0, :]
    source_true_cat_labels = categorical_true_lp[true_dc == 0, :]
    utils.plot_pr_curve(source_pred_cat_labels, source_true_cat_labels, FLAGS.model_dir, LP_CLASSES,
                        name="pr-curve_source_lp.png")

    # Visualize precision-recall curve for target only
    target_pred_cat_labels = pred_lp[true_dc == 1, :]
    target_true_cat_labels = categorical_true_lp[true_dc == 1, :]
    utils.plot_pr_curve(target_pred_cat_labels, target_true_cat_labels, FLAGS.model_dir, LP_CLASSES,
                        name="pr-curve_target_lp.png")

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
