'''
PREPROCESSING DATA

Sample Script Execution:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)

'''

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
   ''' 
   Load Data function
    
    Arguments:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Output:
        df -> Loaded dasa as Pandas DataFrame
   
   ''' 
   messages = pd.read_csv(messages_filepath)
   categories = pd.read_csv(categories_filepath)
   df = messages.merge(categories, on=["id"], how="outer")
   return df 

def clean_data(df):
  ''' 
  
    Clean Data function
    
    Arguments:
        df -> raw DataFrame
    Outputs:
        df -> clean DataFrame
       
  '''  
  categories =  df['categories'].str.split(';', 36, expand=True)
  row = categories.iloc[0]
  category_colnames = row.apply(lambda x:x[0:-2])
  categories.columns = category_colnames
  for column in categories:
    categories[column] = categories[column].str[-1]
    categories[column] = categories[column].astype(int)
    
  df.drop('categories',axis=1,inplace=True)
  df = pd.concat([df,categories],axis=1)
  df.drop_duplicates(inplace=True)
  return df

    
def save_data(df, database_filename):
  '''
    Save Data function
    
    Arguments:
        df -> Clean DataFrame
        database_filename -> database file (.db) destination path
   '''  
  engine = create_engine('sqlite:///'+database_filename)
  df.to_sql('DisasterResponse', engine, index=False,if_exists='replace')
  pass

def main():
    '''
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    '''
    print("Length of system argv:::",len(sys.argv))
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