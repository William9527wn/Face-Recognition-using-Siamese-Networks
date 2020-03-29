from keras.layers import Activation
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential


def build_base_network(input_shape):
    seq = Sequential()

    nb_filters = [6, 12]
    kernel_size = 3

    # Convoution layer 1
    seq.add(Convolution2D(nb_filters[0], kernel_size, kernel_size, input_shape=input_shape,
                            border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2,2)))
    seq.add(Dropout(.25))

    # Convolution Layer 2
    seq.add(Convolution2D(nb_filters[0], kernel_size, kernel_size, input_shape=input_shape,
                            border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
    seq.add(Dropout(.25))

    # Flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq
