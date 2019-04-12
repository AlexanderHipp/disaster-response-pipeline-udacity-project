# Disaster Response Pipeline Project
The project is part of the Data Scientist Udacity nanodegree. Thanks to [figure-eight](https://www.figure-eight.com/) for providing the nessecary data which consists of real messages that have been sent during a specific disaster. The Pipeline is an NLP machine learning algorithm that takes and processes these messages and categorizes them by content. A web application is also being used to conveniently show with category the message is.

### What packages I have used:
This project was developed using:
* flask 1.0.2
* joblib 0.13.1
* nltk 3.4
* pandas 0.24.1 
* plotly 3.5.0
* scikit-learn 0.20.2
* sqlalchemy 1.2.17 

### How to run the project:
1. To set up your database and model, run  following commands in the project's root directory.

    - To run the ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run the ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/disaster_response`

2. Run following command in the app's directory to run the web app.
    `python run.py`.
    
    Please note that for run.py to run properly the database needs to be called `disaster_response.db` and 
    the model `disaster_response.joblib`.

3. Go to http://0.0.0.0:3001/
