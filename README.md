# Disaster Response Pipeline Project

### Project:
Disaster Response Pipeline Project is a project from the Udacity's Data Science Nanodegree. The goal is to create a machine learning pipeline used to categorize disaster events. Based on the categorization an appropriate message can be sent to an agency for relief.
The model is trained on real data provided by [Figure Eight](https://www.figure-eight.com).

Project consists in three parts:

1. ETL Pipeline

The Python script, data/process_data.py is a cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. ML Pipeline

The Python script, train_classifier.py, is a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Web Flask App

Using this app a worker can input a message and visualize the result of the classification.

### Instructions:

1. Install imblearn

Run:
pip install -U imbalanced_learn

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/
