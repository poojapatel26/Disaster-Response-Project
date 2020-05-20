'''
TRAIN CLASSIFIER

How to run this script (Example)
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 
Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
'''

import sys
import pandas as pd
from sqlalchemy import create_engine

import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(message):
    '''
    Tokenize function
    
    Arguments:
        text -> list of text messages (english)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    '''    
     # get list of all urls using regex
    detected_urls = re.findall(url_regex, message)
    
    # replace each url in message string with placeholder
    for url in detected_urls:
        message = message.replace(url, "urlplaceholder")
        
    message=re.sub(r"[^a-zA-Z0-9]"," ",message.lower())
  
    # tokenize text
    tokens = word_tokenize(message)
    
    #removing stopwords from text
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]

    # lemmatize, normalize case, and remove leading/trailing white space
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
   ''' 
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
   ''' 
   model = Pipeline([ ('vect',CountVectorizer(tokenizer=tokenize)),
                      ('tfidf',TfidfTransformer()),
                      ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=15,n_jobs=-1)))
                        ]) 
   return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy)
    
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    """
    #pipeline.fit(X_train,y_train)
    Y_pred=model.predict(X_test)
    
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    
    pass


def save_model(model, model_filepath):
     filename = model_filepath
     pickle.dump(model, open(filename, 'wb'))
     pass
     """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipeline object
        model_filepath -> destination path to save .pkl file
    
    """
   


def main():
    '''
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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