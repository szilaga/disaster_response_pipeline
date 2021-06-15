# Imports
import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads two datasets stored as csv
    and join them together
    retun: dataframe
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id', how='inner')
    
    return df
    


def clean_data(df):
    '''
    This function is transforming the dataset by following steps:
    - category data into multiple columns
    - remove all characters expect of the numbers
    - remove duplicates
    
    return: datframe
    
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True) #<class 'pandas.core.series.Series'>
    
    
    row = categories.iloc[0]
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda x: x[:-2], row))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype("str").str.replace(r'[a-zA-Z_-]','',regex=True)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1, join="inner")
    
    # drop duplicates
    df = df[~df.duplicated() == True]

    # drop invalid values in 'related'
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    '''
    This function is saving the data into a the disaster db
    
    '''
    database = str('sqlite:///'+ database_filename)
    engine = create_engine(database)
    df.to_sql('disaster_clean', engine, index=False, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()