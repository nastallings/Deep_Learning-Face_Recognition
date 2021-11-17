import numpy as np
import Data_Generation as dg
import Model as m
import matplotlib.pyplot as plt
import cv2
import os

# Generate Data
dg.generate_data()

xData = []
yData = []
for image in os.listdir("Data/nstallings"):
    img = cv2.imread("Data/nstallings/" + str(image))
    new_img = cv2.resize(img, (180, 100))
    xData.append(new_img)
    yData.append('nstallings')

splitValue = 0.8
n = int(len(xData)*splitValue)
xDataTrain = xData[:n]
yDataTrain = yData[:n]
xDataTest = xData[n:]
yDataTest = yData[n:]

xDataTrain = np.array(xDataTrain)
yDataTrain = np.array(yDataTrain)
xDataTest = np.array(xDataTest)
yDataTest = np.array(yDataTest)

trainedModel = m.Train_Model(xDataTrain, yDataTrain, 200, 10, 1)
history = m.Test_Model(trainedModel, xDataTest, yDataTest)

# img = cv2.imread("Data/nstallings/nstallings_image_5.png")
# image = cv2.resize(img, (180,100))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()