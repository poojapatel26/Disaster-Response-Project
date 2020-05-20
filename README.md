# Disaster-Response-Project

![](Screenshots/Intro.png)

## Table of Contents

1. [Description](https://github.com/poojapatel26/Disaster-Response-Project#description)
2. [Getting Started](https://github.com/poojapatel26/Disaster-Response-Project#getting-started)
   1. [Dependencies](https://github.com/poojapatel26/Disaster-Response-Project#dependencies)
   2. [Installing](https://github.com/poojapatel26/Disaster-Response-Project#installing)
   3. [Executing Program](https://github.com/poojapatel26/Disaster-Response-Project#executing-program)
   4. [Additional Material](https://github.com/poojapatel26/Disaster-Response-Project#additional-material)	
3. [Licensing, Authors, Acknowledgements](https://github.com/poojapatel26/Disaster-Response-Project#licensing-authors-acknowledgements)
4. [Screenshots](https://github.com/poojapatel26/Disaster-Response-Project#screenshots)
 
## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

 1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
 2. Machine Learning Pipeline to train a model able to classify text message in categories
 3. Web App to show model results in real time.
 
## Getting Started

### Dependencies

* Python 3* 
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly 
 
### Installing

Clone this GIT repository: 
```
https://github.com/poojapatel26/Disaster-Response-Project.git
```
### Executing Program:

1. Run the following commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database ``` python data/process_data.py data/disaster_messages.csv   data/disaster_categories.csv data/DisasterResponse.db ```
   - To run ML pipeline that trains classifier and saves ``` python models/train_classifier.py data/DisasterResponse.db       models/classifier.pkl```
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/

### Additional Material

In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:

  1. ETL Preparation Notebook: learn everything about the implemented ETL pipeline
  2. ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn
  
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section. In this case, it is warmly recommended to use a Linux machine to run Grid Search, especially if you are going to try a large combination of parameters. Using a standard desktop/laptop (4 CPUs, RAM 8Gb or above) it may take several hours to complete.

## Licensing, Authors, Acknowledgements

  * [Udacity](https://www.udacity.com/) for providing such a Amazing project
  * [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model

## Screenshots

1. This is a example of a message you can enter to test Machine Learnning Model Performance, it will categorize your        message:

![](Screenshots/Sample%20Input.png)

2. After clicking on **Classify Message**, you can see the categories which the message belongs to highlighted in green:

![](Screenshots/Sample%20Output.png)


3. The main page shows some graphs about training dataset, provided by Figure Eight:

![](Screenshots/Main%20Page%201.png)

  





   
