import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import tensorflow.keras.backend as tfback

# Sets up tensorflow GPU
tf.compat.v1.disable_eager_execution()


def _get_available_gpus():
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus


class Model:
    """
    The Model Class for face recognition. Deals with creating, training, testing, and displaying the model
    """
    def __init__(self):
        self.model = None

    def Create_Model(self, numPeople):
        """
        Creates the model
        @:param numPeople: number of participants (classes) for the model.
        """
        model = Sequential()
        model.add(Convolution2D(32, 5, 5, input_shape=(192, 168, 3), activation='relu'))
        model.add(Convolution2D(32, 5, 5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(.25))

        model.add(Convolution2D(64, 5, 5, activation='relu', padding='same'))
        model.add(Convolution2D(64, 5, 5, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(.25))

        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(.50))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(.50))
        model.add(Dense(numPeople, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def Train_Model(self, xData, yData, xDataVal, yDataVal, bs, ep):
        """
        Trains the model
        @:param xData: the input images to be classified
        @:param yData: the correct classes corresponding to the xData
        @:param xDataVal: the input test images for validation
        @:param yDataVal: the correct test classes corresponding to the xDataVal for validation
        @:param bs: the batch size for training
        @:param ep: the number of epochs for training
        """
        history = self.model.fit(xData, yData, batch_size=bs, epochs=ep, validation_data=(xDataVal, yDataVal))
        return history

    def Test_Model(self, xTest, yTest):
        """
        Tests the model
        @:param xTest: the input images to tested
        @:param yTest: the correct classes corresponding to the xData
        """
        return self.model.evaluate(xTest, yTest)

    def Display_Model(self, inputImages, outputLabels):
        for index, image in enumerate(inputImages):
            predict_image = []
            predict_image.append(image)
            output = self.model.predict(np.array(predict_image))
            guess = np.argmax(output)
            actual = np.argmax(outputLabels[index])

            if actual == 0:
                plt.title("Image of Nathan Stallings")
                if guess == 0:
                    plt.suptitle("Model Guess: Nathan Stallings")
                else:
                    plt.suptitle("Model Guess: Yale Subject " + str(guess))
            else:
                plt.title("Image of Yale Subject " + str(actual))
                if guess == 0:
                    plt.suptitle("Model Guess: Nathan Stallings")
                else:
                    plt.suptitle("Model Guess: Yale Subject " + str(guess))
            plt.imshow(image)
            plt.show()

    def Save_Model(self, modelName, modelWeights):
        """
        Saves the model to a JSON file
        @:param modelName: the file name for the model
        @:param modelWeights: the file name for the model weights
        """
        model_json = self.model.to_json()
        with open(modelName, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(modelWeights)
        print("Saved model to disk")

    def Load_Model(self, modelName, modelWeights):
        """
        Loads the model from a JSON file
        @:param modelName: the file name for the model
        @:param modelWeights: the file name for the model weights
        """
        try:
            json_file = open(modelName, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(modelWeights)
            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model = loaded_model
            print("Loaded model from disk")
            return 1
        except:
            return 0
