import sys
import pandas as pd
from sqlalchemy import create_engine


def clean_filepath(filepath):
    """
    :description: Takes a file path and it returns it without file extension, it also gives the name of the table.
    """
    if filepath.endswith('.db'):
        clean_path = filepath[:-3]
    else:
        clean_path = filepath
    table_name = clean_path.split('/')[-1]
    return clean_path, table_name


def load_data(messages_filepath, categories_filepath):
    """
    :description: Loads messages.csv and categories.csv files using file paths, and merges dataframes into a singular dataframe.
    :param messages_filepath: string, messages data, with columns: id, message, original and genre
    :param categories_filepath: string, categories data, with columns: id, categories. categories is of the format:
    <categorie_name_1>-<{1, 0}>;<categorie_name_2>-<{1, 0}>;...
    :return: pd.DataFrame, returns a dataframe with the data joined on 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_bad_categories(df):
    """
    :description: Removes categories with one only value from a dataframe, returns the clean dataframe and prints a list of the
    removed columns.
    """
    bad_columns = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, axis=1, inplace=True)
            bad_columns.append(col)
    if bad_columns:
        print('The following categories take only one value and were removed: {}'.format(','.join(bad_columns)))
    return df


def clean_data(df):
    """
    :description: Cleanes the pd.DataFrame from the function load_data and expands the categories column into several columns with
    numeric values and drops Duplicate Rows
    :param df: pd.DataFrame, dataframe from load_data
    :return: pd.DataFrame, cleaned dataframe
    """
    categories = df.categories.str.split(';', expand=True)

    row = categories.iloc[0, :]
    category_colnames = row.str.split('-', expand=True)[0].values
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.split('-', expand=True)[1]
        categories[column] = pd.to_numeric(categories[column])
    categories = clean_bad_categories(categories)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    :description: Saves Cleaned DataFrame as SQLite Database.
    :param df: pd.DataFrame, dataframe to load to the database
    :param database_filename: str, valid file name of the database (no .db ending)
    :return: None
    """
    db_name, table_name = clean_filepath(database_filename)
    engine = create_engine('sqlite:///{}.db'.format(db_name))
    try:
        df.to_sql(table_name, engine, if_exists='fail', index=False)
    except ValueError:
        print('That sqlite database already exists')
    return None


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
