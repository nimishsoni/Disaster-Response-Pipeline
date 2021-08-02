'''Prcesses and cleans text messages and load them to a database'''
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Function to load the messages and categories data (in CSVs) from specified
    filepath and merges the two dataset based on id columns'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    messages_categories = messages.merge(categories, how='left', on='id')
    return messages_categories


def clean_data(messages_categories):
    '''Function to clean the data. Steps: clean categories column data to numeric,
    extract respective column names, create separate category columns and
    drop duplicate rows'''
    categories = messages_categories.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = []
    for cols in row:
        category_colnames.append(cols.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    categories.replace(2, 1, inplace=True)
    messages_categories.drop('categories', axis=1, inplace=True)
    messages_categories = pd.concat([messages_categories, categories], axis=1)
    messages_categories.drop_duplicates(inplace=True)
    return messages_categories


def save_data(messages_categories, database_filename):
    '''Function to save data to a table in SQL database'''
    engine = create_engine('sqlite:///'+ database_filename)
    messages_categories.to_sql('disaster_messages', engine, index=False, if_exists='replace')


def main():
    '''Main function which loads, cleans and saves the data'''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        messages_categories = clean_data(messages_categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(messages_categories, database_filepath)
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