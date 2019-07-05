from keras.layers import Dense, Conv2D, Activation, Dropout
from keras.layers import Input, Flatten, MaxPooling2D
from keras.models import Model

from flipGradientTF import GradientReversal


def dann_mnist(img_width, img_height, img_channels, output_dim):
    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x = Conv2D(32, (5, 5), kernel_initializer='random_normal')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(48, (5, 5), kernel_initializer='random_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    features = Flatten()(x)

    # Labels
    xl = Dense(100, kernel_initializer='random_normal')(features)
    xl = Activation('relu')(xl)
    xl = Dropout(0.5)(xl)
    xl = Dense(100, kernel_initializer='random_normal')(xl)
    xl = Activation('relu')(xl)
    xl = Dropout(0.5)(xl)
    xl = Dense(output_dim, kernel_initializer='random_normal')(xl)
    labels = Activation('softmax')(xl)

    # Domain
    xd = GradientReversal(1.0)(features)
    xd = Dense(100, kernel_initializer='random_normal')(xd)
    xd = Activation('relu')(xd)
    xd = Dropout(0.5)(xd)
    xd = Dense(1, kernel_initializer='random_normal')(xd)
    domains = Activation('sigmoid')(xd)

    # Define adversarial model
    model = Model(inputs=[img_input], outputs=[labels, domains])
    print(model.summary())

    return model
