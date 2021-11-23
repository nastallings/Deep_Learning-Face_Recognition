import cv2
import os
import numpy as np


def get_display_data(maxNumParticipants):
    """
    Gets the display data from the files
    @:param maxNumParticipants: the number of participants (classes) for classification
    """
    xDisplayData = []
    yDisplayData = []

    for image in os.listdir("Display_Data"):
        img = cv2.imread("Display_Data/" + str(image))
        xDisplayData.append(img)
        y = np.zeros(maxNumParticipants, dtype=int)
        if "nstallings" in str(image):
            y[0] = 1
        else:
            start = image.index("B") + len("B")
            end = image.index("_", start)
            index = int(image[start:end])
            y[index] = 1
        yDisplayData.append(y)

    # Convert to np arrays
    xData = np.asarray(xDisplayData)
    yData = np.array(yDisplayData)

    # Shuffle Data
    indices = np.arange(xData.shape[0])
    np.random.shuffle(indices)
    xData = xData[indices]
    yData = yData[indices]

    return xData, yData


def get_test_data(maxNumParticipants, trainingTestingSplit):
    """
    Gets the data from the files
    @:param maxNumParticipants: the number of participants (classes) for classification
    @:param trainingTestingSplit: the percent of data to be used for training vs testing
    """
    xData = []
    yData = []

    # Get Yale Data
    for dir in os.listdir("Data/CroppedYale"):
        number = int(dir[-2:])
        # Only use 7 people
        if number > maxNumParticipants - 1:
            break

        for image in os.listdir("Data/CroppedYale/" + str(dir)):
            if ".pgm" in str(image):
                img = cv2.imread("Data/CroppedYale/" + str(dir) + "/" + str(image))
                xData.append(img)
                y = np.zeros(maxNumParticipants, dtype=int)
                y[number] = 1
                yData.append(y)

    # Get my Data
    for image in os.listdir("Data/nstallings"):
        img = cv2.imread("Data/nstallings/" + str(image))
        xData.append(img)
        y = np.zeros(maxNumParticipants, dtype=int)
        y[0] = 1
        yData.append(y)

    # Convert to np arrays
    xData = np.asarray(xData)
    yData = np.array(yData)

    # Shuffle Data
    indices = np.arange(xData.shape[0])
    np.random.shuffle(indices)
    xData = xData[indices]
    yData = yData[indices]

    # Split data into training and testing
    n = int(len(xData) * trainingTestingSplit)
    xDataTrain = xData[:n]
    yDataTrain = yData[:n]
    xDataTest = xData[n:]
    yDataTest = yData[n:]

    return xDataTrain, yDataTrain, xDataTest, yDataTest


def generate_data(perform):
    """
    Generates data from the files
    @:param perform: a boolean value, if true then generate the data
    """
    index = 0
    if perform:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        for video in os.listdir("Data/Videos"):
            index += 1
            vidcap = cv2.VideoCapture("Data/Videos/" + str(video))
            hasFrames = True
            frameNum = 0
            while hasFrames:
                hasFrames, image = vidcap.read()
                if not hasFrames or index % 13 == 0:
                    break

                # Help randomize the photos
                elif frameNum % 17 != 0:
                    frameNum += 1
                    continue
                else:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Detect the faces
                    face = face_cascade.detectMultiScale(gray, 1.1, 5)
                    if len(face) != 1:
                        continue
                    else:
                        x0 = face[0][0]
                        y0 = face[0][1]
                        x1 = face[0][0] + face[0][2]
                        y1 = face[0][1] + face[0][3]
                        image_face = gray[y0:y1, x0:x1]
                        image_face = cv2.resize(image_face, (168, 192), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join("Data/nstallings", "nstallings_image_%d.png" % index), image_face)
                        index += 1
                        frameNum += 1
