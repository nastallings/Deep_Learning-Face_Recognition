# Deep_Learning-Face_Recognition
This is my final submission for Homework 7-8: Face Recognition for Deep Learning for advanced Perception.
The code can be found at: https://github.com/nastallings/Deep_Learning-Face_Recognition

The report documenting the approach and progress can be found in the file Final_Report.pdf.
This project used the cropped Yale B dataset (as provided in class) and videos taken by my phone.

Model.py is a class that contains the model and incorporates training, testing, and displaying the model.
Data_Generation.py generates the data from the Data folder and the Display_Data folder.

Since the model takes a while to run, and uploading the entire data set used to train the model would be unpractical, 
the model has been pre-trained and saved into two files. Face_Recognition_Model.json contains the saved model and 
Face_Recognition_Weights.h5 contains the model weights. A smaller data set called Display_Data is uploaded so the model
can be tested and run. This file contains well lit pictures of 10 faces for the model to differentiate between.

main.py is what needs to be run for the program to work. It will look for a model to load and create one if a model 
isn't found. If the model needs to be created, a dataset will be needed (this should not happen if properly run). This 
file contains some parameters that need to be set as-is to ensure the program runs correctly:

		maxNumParticipants = 10
		model_JSON = "Face_Recognition_Model.json"
		model_Weights = "Face_Recognition_Weights.h5"
		generateData = False
		haarcascade_frontalface_default.xml needs to be in the directory 

Running main.py will display the models CNN error and a series of images from the Display_Data image set. These images 
have the model's prediction for who it is and the actual value associated to the image. Two examples of this are saved 
as nstallings.png and other.png. 
