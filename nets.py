import tensorflow as tf
import keras.backend as K
from keras.engine import Layer
from keras.layers import Dense, Conv2D, Activation, Dropout
from keras.layers import Input, Flatten, MaxPooling2D
from keras.models import Model


def reverse_gradient(X, hp_lambda):
    """
    Flips the sign of the incoming gradient during training.
    """
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    """
    Layer that flips the sign of gradient during training.
    """
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = K.variable(hp_lambda)  # K variable to be modified during training via a callback

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, K.get_value(self.hp_lambda))

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': K.get_value(self.hp_lambda)}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def dann_mnist(img_width, img_height, img_channels, output_dim):
    """
    Domain Adaptation Neural Network (DANN) for domain adaptation between real and synthetic MNIST datasets.
    :param img_width: target image width.
    :param img_height: target image height.
    :param img_channels: target image channels.
    :param output_dim: output dimension of the label predictor (number of classes).
    :return: a Model instance.
    """
    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    # Feature extractor
    x = Conv2D(32, (5, 5), kernel_initializer='random_normal')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(48, (5, 5), kernel_initializer='random_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    features = Flatten()(x)

    # Label predictor
    xl = Dense(100, kernel_initializer='random_normal')(features)
    xl = Activation('relu')(xl)
    xl = Dropout(0.5)(xl)
    xl = Dense(100, kernel_initializer='random_normal')(xl)
    xl = Activation('relu')(xl)
    xl = Dropout(0.5)(xl)
    xl = Dense(output_dim, kernel_initializer='random_normal')(xl)
    lp = Activation('softmax')(xl)

    # Gradient Reversal Layer
    xg = GradientReversal(1.0)(features)

    # Domain classifier
    xd = Dense(100, kernel_initializer='random_normal')(xg)
    xd = Activation('relu')(xd)
    xd = Dropout(0.5)(xd)
    xd = Dense(1, kernel_initializer='random_normal')(xd)
    dc = Activation('sigmoid')(xd)

    # Define adversarial model
    model = Model(inputs=[img_input], outputs=[lp, dc])
    print(model.summary())

    return model
