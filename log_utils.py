import os
import keras
from keras import backend as K

import logz
from nets import GradientReversal


class MyCallback(keras.callbacks.Callback):

    def __init__(self, filepath, period, batch_size, total_steps):
        """
        Customized callback class.
        :param filepath: path to save model.
        :param period: frequency in epochs to save models.
        :param batch_size: batch size (considering source and target).
        :param total_steps: number of total training steps.
        """
        self.filepath = filepath
        self.period = period
        self.batch_size = batch_size
        self.total_steps = total_steps

    def on_batch_begin(self, batch, logs=None):
        # Learning rate schedule
        K.set_value(self.model.optimizer.lr, 0.01 / ((1. + 10. * float(batch) / self.total_steps) ** 0.75))

        # Schedule for domain adaptation parameter
        # for layer in self.model.layers:
        #     if isinstance(layer, GradientReversal):
        #         K.set_value(layer.hp_lambda, 2./(1.+ np.exp(-10.*float(batch)/self.total_steps))-1.)
        #         break

    def on_epoch_end(self, epoch, logs={}):
        # Save logs of training loss
        logz.log_tabular('train_lp', logs.get('activation_5_loss'))
        logz.log_tabular('train_dc', logs.get('activation_7_loss'))

        # Save logs of validation loss
        logz.log_tabular('val_lp', logs.get('val_activation_5_loss'))
        logz.log_tabular('val_dc', logs.get('val_activation_7_loss'))
        logz.dump_tabular()

        # Save model every 'period' epochs
        if (epoch + 1) % self.period == 0:
            filename = os.path.join(self.filepath, 'model_weights_{epoch:03d}.h5')
            print("Saved model at {}".format(filename))
            self.model.save_weights(filename, overwrite=True)

        # # Save model with the highest validation loss from epoch 10
        # if epoch > 10:
        #     #fname = self.filepath + '/dom_weights_' + str(epoch) + '.h5'
        #     fname = os.path.join(self.filepath, 'dc_weights_{epoch:03d}.h5')
        #     current_loss = logs.get('val_activation_7_loss')
        #     if current_loss > K.get_value(self.model.dom_loss):
        #         K.set_value(self.model.dom_loss, current_loss)
        #         self.model.save_weights(fname, overwrite=True)
