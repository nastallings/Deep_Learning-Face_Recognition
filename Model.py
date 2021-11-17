from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import tensorflow.keras.backend as tfback
tf.compat.v1.disable_eager_execution()


def _get_available_gpus():
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


def Model(num_people):
    # create model
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, input_shape=(1, 180, 100), data_format="channels_first", activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.25))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.25))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.50))
    model.add(Dense(num_people, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def Train_Model(xData, yData, bs, ep, numPeople):
    model = Model(numPeople)
    return model.fit(xData, yData, batch_size=bs, epochs=ep)


def Test_Model(model, xTest, yTest):
    return model.evaluate(xTest, yTest)

