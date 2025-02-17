import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, \
                         Convolution1D, Convolution2D, MaxPooling2D, \
                         Flatten, LSTM, BatchNormalization, \
                         Input, MaxPool1D, GlobalMaxPool1D


def build_deep_dense(n_features=193, n_classes=8):
    model = Sequential()
    model.add(Dense(100, input_dim=n_features))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1200, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(1200, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1400, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    return model


def build_conv_seq(input_shape, n_classes=8):
    #   Based on http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
    #  default input shape is (bands, frames, num_channels)
    model = Sequential()
    f_size = 1

    # first layer has 48 convolution filters
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same',
                            input_shape=input_shape))
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten output into a single dimension
    model.add(Flatten())

    # then a fully connected NN layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # finally, an output layer with one node per class
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    return model


def build_conv_spectre():
    pass


def build_raw_1dconv_kaggle(input_length, n_classes=8):

    inp = Input(shape=(input_length, 1))
    x = Convolution1D(16, 9, activation='relu', padding="valid")(inp)
    x = Convolution1D(16, 9, activation='relu', padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation='relu', padding="valid")(x)
    x = Convolution1D(32, 3, activation='relu', padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation='relu', padding="valid")(x)
    x = Convolution1D(32, 3, activation='relu', padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(256, 3, activation='relu', padding="valid")(x)
    x = Convolution1D(256, 3, activation='relu', padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dense(1028, activation='relu')(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model


def build_lstm_seq(timesteps=20, data_dim=41, n_classes=8):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    # returns a sequence of vectors of dimension 256
    model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(Dropout(0.2))

    # return a single vector of dimension 128
    model.add(LSTM(128))
    model.add(Dropout(0.2))

    # apply softmax to output
    model.add(Dense(n_classes, activation='softmax'))
    return model
