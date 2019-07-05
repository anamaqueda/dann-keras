import gflags
import numpy as np
import os
import sys
from sklearn import metrics

from keras import backend as K

import utils
import data_utils
from common_flags import FLAGS
from flipGradientTF import GradientReversal


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Constants
TEST_PHASE = 0
LAB_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DOM_CLASSES = ['source-real', 'target-syn']


def _main():

    # Set testing mode (dropout/batch normalization)
    K.set_learning_phase(TEST_PHASE)
    
    # Output dimension (3 classes: green, red, yellow)
    output_dim = 10

    # Generate testing data
    test_data = data_utils.DataLoader(FLAGS.test_dir, output_dim, FLAGS.img_mode, (FLAGS.img_height, FLAGS.img_width))
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = data_utils.batch_generator(test_data, output_dim, FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.json_to_model(json_model_path, {'GradientReversal': GradientReversal})

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        raise ValueError("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())

    # Get predictions and ground truth
    n_samples = test_data.num_source + test_data.num_target
    nb_batches = int(np.ceil(n_samples / (FLAGS.batch_size*2)))
    # Get predicted probabilities and ground truth
    labels, domains, categorical_true_labels, true_domains = utils.compute_predictions_and_gt(model, test_generator, nb_batches, verbose=1)

    # Convert ground truth of labels and domains to an appropriate format
    true_labels = np.argmax(categorical_true_labels, axis=-1)

    # Convert predictions of labels and domains to an appropriate format
    pred_labels = np.argmax(labels, axis=-1)
    pred_domains = domains.T[0]
    pred_bin_domains = np.zeros_like(pred_domains)
    pred_bin_domains[pred_domains >= 0.5] = 1

    # Save predictions and ground_truth as a dictionary
    labels_dict = {'pred_labels': labels.tolist(),
                   'pred_domains': pred_domains.tolist(),
                   'true_labels': true_labels.tolist(),
                   'true_domains': true_domains.tolist()}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir, 'predictions_and_gt.json'))

    # Visualize domain confusion matrix
    utils.plot_confusion_matrix(true_domains, pred_bin_domains, DOM_CLASSES, FLAGS.experiment_rootdir, normalize=True,
                                name="confusion_domains.png")
    print('Domain accuracy =', metrics.accuracy_score(true_domains, pred_bin_domains))

    # Visualize label confusion matrix for source only
    source_pred_labels = pred_labels[true_domains == 0]
    source_true_labels = true_labels[true_domains == 0]
    utils.plot_confusion_matrix(source_true_labels, source_pred_labels, LAB_CLASSES, FLAGS.experiment_rootdir,
                                normalize=True, name="confusion_source_labels.png")
    print('Label accuracy on source data =', metrics.accuracy_score(source_true_labels, source_pred_labels))

    # Visualize label confusion matrix for target only
    target_pred_labels = pred_labels[true_domains == 1]
    target_true_labels = true_labels[true_domains == 1]
    utils.plot_confusion_matrix(target_true_labels, target_pred_labels, LAB_CLASSES, FLAGS.experiment_rootdir,
                                normalize=True, name="confusion_target_labels.png")
    print('Label accuracy on target data =', metrics.accuracy_score(target_true_labels, target_pred_labels))

    # Visualize precision-recall curve for source only
    source_pred_cat_labels = labels[true_domains == 0, :]
    source_true_cat_labels = categorical_true_labels[true_domains == 0, :]
    utils.plot_custom_pr_curve(source_pred_cat_labels, source_true_cat_labels, FLAGS.experiment_rootdir,
                               LAB_CLASSES, name="pr-curve_source_labels.png")

    # Visualize precision-recall curve for target only
    target_pred_cat_labels = labels[true_domains == 1, :]
    target_true_cat_labels = categorical_true_labels[true_domains == 1, :]
    utils.plot_custom_pr_curve(target_pred_cat_labels, target_true_cat_labels, FLAGS.experiment_rootdir,
                               LAB_CLASSES, name="pr-curve_target_labels.png")

                                               
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
