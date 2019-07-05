import numpy as np
import json
import os
import matplotlib.pyplot as plt
import itertools
import shutil
import pickle
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from keras.utils.generic_utils import Progbar
from keras.models import model_from_json


COLORS = ['g', 'r', 'y', 'c', 'k', 'm', 'pink', 'darkgreen', 'orange', 'beige']
DECISION_THR = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.15, 0.05]


def compute_predictions_and_gt(model, generator, steps, verbose=0):
    """
    Generate predictions and associated ground truth for the input samples from a data generator. The generator should
    return the same kind of data as accepted by 'predict_on_batch'.

    Function adapted from Keras 'predict_generator'.

    :param model: Model instance containing a trained model.
    :param generator: Generator yielding batches of input samples.
    :param steps: Total number of steps (batches of samples) to yield from 'generator' before stopping.
    :param verbose: verbosity mode, 0 or 1.
    :return: numpy arrays of predictions and associated ground truth.
    """
    steps_done = 0
    all_lp_pred = []  # LP predictions
    all_lp_gt = []  # LP ground truth
    all_dc_pred = [] # DC predictions
    all_dc_gt = [] # DC ground truth

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_label = generator_output
            elif len(generator_output) == 3:
                x, gt_label, _ = generator_output
            else:
                raise ValueError('output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        # Forward pass on testing data
        outs = model.predict_on_batch(x)

        all_lp_pred = np.concatenate((all_lp_pred, np.squeeze(outs[0])), axis=0)
        all_lp_gt = np.concatenate((all_lp_gt, gt_label[0]), axis=0)
        all_dc_pred = np.concatenate((all_dc_pred, np.squeeze(outs[1])), axis=0)
        all_dc_gt = np.concatenate((all_dc_gt, gt_label[1]), axis=0)

        steps_done += 1
        if verbose == 1:
            progbar.update(steps_done)

    return all_lp_pred, all_dc_pred, all_lp_gt, all_dc_gt


def model_to_json(model, json_model_path):
    """
    Serialize a model into JSON.
    :param model: Model instance containing a trained model.
    :param json_model_path: Directory where saving the model in JSON format.
    """
    model_json = model.to_json()
    with open(json_model_path, "w") as f:
        f.write(model_json)


def json_to_model(json_model_path, custom_objects=None):
    """
    Serialize JSON into a model.
    :param json_model_path: Directory where JSON model is.
    :return: A model instance containing a trained model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    return model


def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in JSON format.
    :param dictionary: Dictionary to be saved.
    :param fname: Name of the JSON file.
    """
    with open(fname, "w") as f:
        json.dump(dictionary, f)
        print("Written file {}".format(fname))
        
        
def plot_loss(path_to_log):
    """
    Read log life and plot losses.
    :param path_to_log: Directory where log file is.
    """
    # Read log file
    log_file = os.path.join(path_to_log, "log.txt")
    try:
        log = np.genfromtxt(log_file, delimiter='\t', dtype=None, names=True)
    except:
        raise IOError("Log file not found")

    train_loss = log['label_predictor']
    val_loss = log['domain_classifier']
    timesteps = list(range(train_loss.shape[0]))
    
    # Plot losses
    plt.plot(timesteps, train_loss, 'r--', timesteps, val_loss, 'b--')
    plt.legend(["Label predictor", "Domain classifier"])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(path_to_log, "log.png"))
    
    
def plot_confusion_matrix(real_labels, pred_labels, classes, output_path, normalize=True, name="confusion.png"):
    """
    Plot confusion matrix computed from predicted and real labels
    :param real_labels: List of real labels
    :param pred_labels: List of predicted labels for a given threshold (specified in FLAGS.classification_thr)
    :param classes: Names of the classes
    :param output_path: Directory where saving the confusion matrix as PNG format
    :param normalize: Whether to normalize values between 0 and 1
    """
    # Generate confusion matrix
    cm = confusion_matrix(real_labels, pred_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float('%.3f'%(cm[i, j])), horizontalalignment="center")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, name))


def plot_pr_curve(pred_probabilities, true_labels, classes, output_path):
    """
    Compute precision and recall for a set of thresholds.
    :param pred_probabilities: Predicted probabilities
    :param true_labels: Real labels
    :param output_path: Directory where saving the curve in PNG format
    Otherwise, they are computed for a set of threshold values.
    :return: Precision-recall curves for every category
    """

    # Precision-recall figure
    plt.figure(figsize=(10, 5))
    plt.title('Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.05)

    for i in range(true_labels.shape[1]):
        precision, recall, thresholds = precision_recall_curve(true_labels[:, i], pred_probabilities[:, i])
        average_precision = average_precision_score(true_labels[:, i], pred_probabilities[:, i])
        print('Average precision score for ' + CLASSES[i] + ' class: {0:0.2f}'.format(average_precision))

        # Plot per category
        plt.plot(recall, precision, '-o', color=COLORS[i], label=classes[i])
        plt.tight_layout()

    plt.legend()
    plt.savefig(os.path.join(output_path, 'PR-curve.png'))


def plot_custom_pr_curve(pred_probabilities, true_labels, output_path, classes, name="pr-curve.png"):
    """
    Compute precision and recall for a set of thresholds.
    :param pred_probabilities: Predicted probabilities
    :param true_labels: Real labels
    :param output_path: Directory where saving the curve in PNG format
    Otherwise, they are computed for a set of threshold values.
    :return: Precision-recall curves for every category
    """

    # Precision-recall figure
    plt.figure(figsize=(10, 5))
    plt.title('Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, 1.05)

    for cat in range(true_labels.shape[1]):

        probabilities = pred_probabilities[:, cat].tolist()
        labels = true_labels[:, cat].tolist()

        precision = []
        recall = []

        for thr in DECISION_THR:
            tp = len([i for i, prob in enumerate(probabilities) if (prob >= thr and labels[i] == 1)])
            fp = len([i for i, prob in enumerate(probabilities) if (prob >= thr and labels[i] == 0)])
            fn = len([i for i, prob in enumerate(probabilities) if (prob < thr and labels[i] == 1)])
            tn = len([i for i, prob in enumerate(probabilities) if (prob < thr and labels[i] == 0)])

            if tp + fp == 0:
                continue
            if tp + fn == 0:
                continue

            P = tp / (tp + fp)
            R = tp / (tp + fn)

            precision.append(P)
            recall.append(R)

        # Plot per category
        plt.plot(recall, precision, '-o', color=COLORS[cat], label=classes[cat])
        plt.tight_layout()

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, name))


def compute_errors(directory, filenames, pred_probabilities, classes, output_path):
    # Compute misclassified samples for yellow category only
    pred_labels = np.argmax(pred_probabilities, axis=1)
    pred_yellow_labels = pred_labels[-100:]
    filenames = filenames[-100:]

    # Save correctly classified yellow samples
    correct_path = os.path.join(output_path, 'correct')
    if not os.path.isdir(correct_path):
        os.makedirs(correct_path)

    correct_samples = []
    correct_categories = []
    incorrect_samples = []
    incorrect_categories = []

    # Save misclassified yellow samples
    incorrect_path = os.path.join(output_path, 'incorrect')
    if not os.path.isdir(incorrect_path):
        os.makedirs(incorrect_path)

    tp = np.where(pred_yellow_labels == 2)[0].tolist()
    for ind in range(0, len(filenames)):
        cat = pred_yellow_labels[ind]
        src = os.path.join(directory, filenames[ind])
        if ind in tp:
            correct_samples.append(src)
            correct_categories.append(classes[cat])
            dst = os.path.join(correct_path, 'pred_' + classes[cat] + '_' + os.path.basename(filenames[ind]))
            shutil.copy2(src, dst)
        else:
            incorrect_samples.append(src)
            incorrect_categories.append(classes[cat])
            dst = os.path.join(incorrect_path, 'pred_' + classes[cat] + '_' + os.path.basename(filenames[ind]))
            shutil.copy2(src, dst)

    # Store correct samples list as binary data stream
    fname_correct = os.path.join(correct_path, 'correct_samples.pkl')
    with open(fname_correct, 'wb') as f_correct:
        pickle.dump([correct_samples, correct_categories], f_correct)

    # Store incorrect samples list as binary data stream
    fname_incorrect = os.path.join(incorrect_path, 'incorrect_samples.pkl')
    with open(fname_incorrect, 'wb') as f_incorrect:
        pickle.dump([incorrect_samples, incorrect_categories], f_incorrect)


def compute_highest_classification_errors(predictions, ground_truth, n_errors=20):
    """
    Compute the 'n_errors' highest errors predicted by the model.
    :param predictions: Predicted probabilities.
    :param ground_truth: Real labels.
    :param n_errors: Number of samples with highest error to be returned.
    :return: highest_errors: Indexes of the samples with highest errors.
    """
    assert np.all(predictions.shape == ground_truth.shape)
    dist = abs(predictions - 1)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors