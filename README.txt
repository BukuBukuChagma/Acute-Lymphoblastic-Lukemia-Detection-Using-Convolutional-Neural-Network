==> Dataset
Dataset filder contains our Dataset. There are three folders to begin with.
	i)   orignalTrain      : We will be using this dataset to train our model
        ii)  orignalValidation : we will be using this dataset for validating our model
        iii) orignalTest       : we will be using this dataset to test how our model preforms when deployed using the Web App


==> Info About all the files in Project

1. The Traning File is for the Image Processing On Traning Data
2. Validation File is for Image Processing On Validation Data
3. Neural Network File is For Traning our Model Using The Processed Traning Data and then Validation using Processed Validation Data
4. Testing File is for Image Processing on the Testing Image That we'll get from user through Web App and then predicting the result
5. main.py is our main file that connects our Web App to our Model. This is the file that should be run.
6. Model folder contains our trained Model which we will load when needed(for pridicting)
7. Dataset Folder contains the Dataset.(I suggest not to mess around with that folder, because it can cause problem in paths configuration)
8. template folder contains our template for our Web App
9. static folder contains data need for our Web App.

Since our model is already trained, there is no need to run the Traning and Validation Data files. Simply Run the Main.py file and 
copy paste the link it provides in console to your browser and you are good to go.

Project Done By:
Muhammed Luqman
Wafiya Sohail
Farhan Shoukat