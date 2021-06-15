# import libraries
import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class StartingNounExtractor(BaseEstimator, TransformerMixin):
    
    def starting_noun(self, text):
        '''
        This function is detecting, if a sentence is starting with a noun 
        at the beginning
        '''
    
        try:
            pos_tags = nltk.pos_tag(tokenize(text)) # returns a list of tokens
            print(pos_tags)
            first_word, first_tag = pos_tags[0] # take first element of token list per sentence

            if first_tag in ['NN', 'NNE']: # check if first element is a verb
                 return True 
            return False
        except: return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        The transform function returns a dataframe 
        whereby each sentence is checked if they start by a verb
        the dataframe contains true and false values
        """
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return pd.DataFrame(X_tagged)
    

def load_data(database_filepath):
    '''
    This function loads data from a database table
    Return x = sentence, y = categories 
    '''
    # load data from database
    database = str('sqlite:///'+ database_filepath)
    engine = create_engine(database)
   
    sqlquery = 'SELECT * FROM disaster_clean'
    df = pd.read_sql_query(sqlquery,con=engine)
    
    # extract message column from dataframe
    X = df['message'].values
    
    # drop unnecessary columns from dataframe
    df.drop(['id','message','original','genre'],axis=1, inplace=True)
    
    # extract 36 categories
    y = df[list(df.columns)].values
    
    return X, y, df.columns


def tokenize(text):
    '''
    The tokenizer function converts sentence in a list of words
    Return list
    '''
    
    #remove unwanted characters:
    text = re.sub(r'[^a-zA-Z0-9]',' ', text)

    #change a sentence into tokens (words)
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w  not in stopwords.words('english')]

    #create object
    lemmatizer = WordNetLemmatizer()

    #intiate empty list
    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
  
    '''
    Creates a pipline with a Multioutput Classifier
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) #KNeighborsClassifier()
    ])
    
    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    This function shows f1 score, precision and recall
    '''    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
   
    


def save_model(model, model_filepath):
    
    '''
    This function is saving the ML model as a pickle file
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
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