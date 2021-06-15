# disaster_response_pipeline
Machine learning pipeline to detect disasters

## Table of contents

- [Quickstart](#quick-start)
- [Instructions](#Instructions)
- [Idea](#idea)
- [What's included](#whats-included)
- [Description](#description)
- [Copyright and license](#copyright-and-license)


# Quickstart
Download the entire Git Repo including the datasets. Execute the script <b>"run.py"</b> in /app to start the webapp.<br/>
Before execution: If you do not already have installed the listed libraies below, please install them in a previous step.

- Install <a href="https://pypi.org/project/pandas/">Pandas:</a> `pip install pandas`
- Install <a href="https://pypi.org/project/numpy/">Numpy:</a> `pip install numpy`
- Install <a href="https://pypi.org/project/matplotlib"/>Plotlib:</a> `pip install matplotlib` 
- Install <a href="https://pypi.org/project/nltk/">nltk:</a> `pip install nltk`
- Install <a href="https://pypi.org/project/SQLAlchemy/">SQLAlchemy:</a> `pip install SQLAlchemy`
- Install <a href="https://pypi.org/project/scikit-learn/">SCIKIT-Learn:</a> `pip install scikit-learn`

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Idea
This projects analyses real world disaster messages via ML workflow and provides one or more categories, of the disaster type


## What's included
```text
disaster_response_pipeline/
├── data/
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── process_data.py
├── models/
|   ├── classifier.pkl
|   ├── train_classifier.py
├── app/
|   ├── run.py
|   ├── templates/
|   |   ├── go.html
|   |   ├── master.html
├── README.MD
```



## Description

#### process_data.py:
This python file is executing the ETL pipeline by open the given csv datasets.
Joining and cleaning the dataset and stores it into a database.

#### train_classifier.py
This pyhton file is loading the dataset, preparing a the ML pipeline and trains the model.

#### run.py:
run pyhton starts the webapp.

## Contributing
If you like this project, please feel free to code. 

## Copyright and license
The code itself is free licensed. All libaries, used in this project are still under the given licence model.
For more information please check the licence @pypi.org for every library listed above.


