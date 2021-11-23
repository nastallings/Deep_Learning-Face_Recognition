import numpy as np
import Data_Generation as dg
import Model as m
import matplotlib.pyplot as plt
import cv2
import os

# Parameters
maxNumParticipants = 10
model_JSON = "Face_Recognition_Model.json"
model_Weights = "Face_Recognition_Weights.h5"
generateData = False

# Load or create model
model = m.Model()
if model.Load_Model(model_JSON, model_Weights) == 1:
    xDisplayData, yDisplayData = dg.get_display_data(maxNumParticipants)
    # Score the model
    score = model.Test_Model(xDisplayData, yDisplayData)
    print("CNN Error: %.2f%%" % (100-score[1]*100))
    # Display the models predictions
    model.Display_Model(xDisplayData, yDisplayData)

else:
    # Generate Data
    dg.generate_data(generateData)
    xDataTrain, yDataTrain, xDataTest, yDataTest = dg.get_test_data(maxNumParticipants, 0.8)
    model.Create_Model(maxNumParticipants)
    # Train the model (xData, yData, batch_size, epochs)
    history = model.Train_Model(xDataTrain, yDataTrain, xDataTest, yDataTest, 1, 150)
    model.Save_Model("Face_Recognition_Model.json", "Face_Recognition_Weights.h5")

    # Plot the data
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(history.history['accuracy'], color='b')
    axs[0].plot(history.history['val_accuracy'], color='r')
    axs[0].title.set_text('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'test'], loc='lower right')

    # summarize history for loss
    axs[1].plot(history.history['loss'], color='b')
    axs[1].plot(history.history['val_loss'], color='r')
    axs[1].title.set_text('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'test'], loc='upper right')

    fig.tight_layout(pad=2.0)
    plt.show()

    # Score the model
    score = model.Test_Model(xDataTest, yDataTest)
    print("CNN Error: %.2f%%" % (100-score[1]*100))


