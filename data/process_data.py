import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath : path to a csv file containing the messages
    categories_filepath : path to a csv file containing the corresponding multi-labels
    
    OUTPUT
    DataFrame merging the information
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages, on='id')
    return df
   

def clean_data(df):
    '''
    INPUT 
    df : DataFrame containing the row data
    OUTPUT
    Cleaned data frame
    '''
    #Split categories into separate category columns
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0]
    categories.columns = [ x.split('-')[0] for x in row]
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if not str(x)=='nan' else x )
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #concatenate back messages and the now cleaned categories
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    
    #remove duplicates
    index_drop = df[df.duplicated(subset=None, keep='first')].index
    df = df.drop(index_drop, axis=0)
    
    #remove empty columns
    df = df.drop(['child_alone'],axis=1)
    
    #correct value
    df['related'] = df['related'].replace(2,1)
    return df


def save_data(df, database_filename):
    '''
    INPUT
    df : DataFrame with th date to be saved in the database_filename
    database_filename : file to be created to saved the data
    
    OUTPUT
    A file with a database structure containing the data in df
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message', engine, index=False, if_exists='replace') 


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