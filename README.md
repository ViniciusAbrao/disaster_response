# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

This project is according to https://www.udacity.com/

"The objective of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages".
There are 36 categories in which the message can be simultaneously classified.

The first step is the ETL pipeline which takes the data from files disaster_categories.csv and disaster_messages.csv.
The steps of lemmatize, normalize case, and remove leading/trailing white space are applied to the texts messages.
The categories columns have binary values of 0 and 1, if the correspondent message is classified (1) or not (0) in each category.
Text and categories are merged in a final pandas data Frame and converted to a SQL database named DisasterResponse.db.

The second step is the ML pipeline. It initially takes the DisasterResponse.db with the clean data.
The Sklearn MultiOutputClassifier is used with the estimator RandomForestClassifier.
The total number of messages in the training set is 26177.
After that, the model is saved as classifier.pkl to be used in the web app.

A new message can be classified in the web app.
At the top of the page, there are links to the Udacity website and the github repo.
The web app shows in the Figures an overview of the dataset used in the training set.
In Figure 1 we can see the number of total occurrences per category.
Figure 2 presents the distribution of data with the counts of 0 and 1 in each category.

### In the github repo the file structure of the project is:

- app
- | - template
- | |- master.html  # main page of web app
- | |- go.html  # classification result page of web app
- |- run.py  # Flask file that runs app

- data
- |- disaster_categories.csv  # data to process 
- |- disaster_messages.csv  # data to process
- |- process_data.py
- |- InsertDatabaseName.db   # database to save clean data to

- models
- |- train_classifier.py
- |- classifier.pkl  # saved model 

- README.md
