# import libraries
import sys
from sqlalchemy import create_engine
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    
    #create engine
    engine = create_engine('sqlite:///'+database_filepath)
    #convert to pandas
    df = pd.read_sql('Select * From DisasterResponse',engine)
    #input and output columns
    X=df['message']
    Y=df[df.columns[2:]]
    
    return X, Y


def tokenize(text):
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    #pipeline model - parameters are not optimized in order to run faster
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline

def optimize_model():
    
    #pipeline model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
         
    #grid serch parameters
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0), #best 0.75
        'tfidf__use_idf': (True, False), #best false
        'clf__estimator__n_estimators': [50, 100, 200], #set to 100
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test):
    #compute y_predicted
    y_pred = model.predict(X_test)
    #compute accuracy
    accuracy = (y_pred == Y_test).mean()
    print('Accuracy: {}'.format(accuracy))
    print('\n')
    #loop over the output columns to compute the f1 score, precision and recall
    columns = list(Y_test)
    y_pred=pd.DataFrame(y_pred,columns=columns)
    for i in columns: 
        print('Column: {}'.format(i))
        print(classification_report(Y_test[i],y_pred[i]))
    return

def save_model(model, model_filepath):
    #open the file name and write the model
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()