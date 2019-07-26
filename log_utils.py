import os
import numpy as np
import keras
from keras import backend as K

from nets import GradientReversal


class MyCallback(keras.callbacks.Callback):

    def __init__(self, model_dir, period, batch_size, total_steps):
        """
        Customized callback class.
        :param model_dir: path to save model.
        :param period: frequency in epochs to save models.
        :param batch_size: batch size (considering source and target).
        :param total_steps: number of total training steps.
        """
        self.model_dir = model_dir
        self.period = period
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.train_logs = dict()  # Dictionary for training losses
        self.val_logs = dict()  # Dictionary for validation losses

    def on_train_begin(self, logs={}):
        # Define training logs
        self.train_logs['lp_loss'] = []  # LP training loss
        self.train_logs['dc_loss'] = []  # DC training loss
        self.train_logs['total_loss'] = []  # Total training loss
        self.train_logs['source_acc'] = []  # Training accuracy for source samples
        self.train_logs['target_acc'] = []  # Training accuracy for target samples
        self.train_logs['dc_acc'] = []  # DC training accuracy

        # Define validation logs
        self.val_logs['lp_loss'] = []  # LP validation loss
        self.val_logs['dc_loss'] = []  # DC validation loss
        self.val_logs['total_loss'] = []  # Total validation loss
        self.val_logs['source_acc'] = []  # Validation accuracy for source samples
        self.val_logs['target_acc'] = []  # Validation accuracy for target samples
        self.val_logs['dc_acc'] = []  # DC validation accuracy

    def on_batch_end(self, batch, logs=None):
        # Count number of steps
        K.set_value(self.model.current_step, (K.get_value(self.model.current_step) + 1.))

    def on_batch_begin(self, batch, logs=None):
        # Learning rate schedule
        K.set_value(self.model.optimizer.lr,
                    0.01 / ((1. + 10. * float(K.get_value(self.model.current_step))/self.total_steps) ** 0.75))

        # Schedule for domain adaptation parameter (not used)
        # for layer in self.model.layers:
        #     if isinstance(layer, GradientReversal):
        #         K.set_value(layer.hp_lambda,
        #                     2./(1.+ np.exp(-10.*float(K.get_value(self.model.current_step))/self.total_steps))-1.)
        #         break

    def on_epoch_end(self, epoch, logs={}):
        # Save training logs
        self.train_logs['lp_loss'].append(logs.get('activation_5_loss'))
        self.train_logs['dc_loss'].append(logs.get('activation_7_loss'))
        self.train_logs['total_loss'].append(logs.get('loss'))
        self.train_logs['source_acc'].append(logs.get('activation_5_custom_acc_source'))
        self.train_logs['target_acc'].append(logs.get('activation_5_custom_acc_target'))
        self.train_logs['dc_acc'].append(logs.get('activation_7_custom_bin_acc'))

        # Save validation logs
        self.val_logs['lp_loss'].append(logs.get('val_activation_5_loss'))
        self.val_logs['dc_loss'].append(logs.get('val_activation_7_loss'))
        self.val_logs['total_loss'].append(logs.get('val_loss'))
        self.val_logs['source_acc'].append(logs.get('val_activation_5_custom_acc_source'))
        self.val_logs['target_acc'].append(logs.get('val_activation_5_custom_acc_target'))
        self.val_logs['dc_acc'].append(logs.get('val_activation_7_custom_bin_acc'))

        # Save model every 'period' epochs
        if (epoch + 1) % self.period == 0:
            filename = os.path.join(self.model_dir, 'model_weights_' + str(epoch).zfill(3) + '.h5')
            print("Saved model at {}".format(filename))
            self.model.save_weights(filename, overwrite=True)
