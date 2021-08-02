# Disaster-Response-Pipeline
Analyze disaster data to build a model for an API that classifies disaster messages in to 36 Categories

## Installations and Quick Start
Install Python Packages: Pandas, Numpy,re, sys, json, sqlite3, sklearn, nltk, sqlalchemy, pickle, Flask, plotly

## Project Motivation
This work is part of Udacity Data Science Nanodegree project requirement. The task here is to analyze disaster data to build a model for an API that classifies disaster related text messages. This categorization aids different departments to take necessary action in the event of a disaster, where people post text messages through social media or news. Thus people can be reached and provided necessary aid as quickly as possible. A web app is created where user can enter a text message which is categorized by the classifier model. 

## Project Details
### ETL Pipeline: 
- Load messages and categories data sets 
- Inspect and clean the data
- Store it in SQL DB 

### ML Pipeline: 
- Load and Tokenize the data
- Build ML Pipeline consiting of feature preparation and classifier along with grid search to find the best parameters
- Train the pipeline model
- Test the model using F1 score, Precision and Recall

### Web App
-  The web app enables the user to enter a text message, and then uses the model to classify the message and provide the output

## Files
README.md: read me file
- ETL Pipeline Preparation.ipynb:  ETL pipeline preparation code in Jupyter notebook
- ML Pipeline Preparation.ipynb: ML pipeline preparation code in Jupyter notebook
Folders
	- \data
		- disaster_categories.csv: categories dataset
		- disaster_messages.csv: messages dataset
		- DisasterResponse.db: disaster response database
		- process_data.py: ETL process
	- \models
		- train_classifier.py: classification code
  - \app
		- run.py: file to run the app
	- \templates
		- master.html: main page 
		- go.html: result web page

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/DisasterResponse_classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Author
Nimish Soni
