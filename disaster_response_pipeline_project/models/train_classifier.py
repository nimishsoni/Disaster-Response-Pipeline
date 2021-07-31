''' Code to create ML pipeline for Disaster Management text classification'''
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''Function to load the Text messages data from SQL database. It further segregates
    the actual messages as input variable (X) and category variables as target variable(Y)'''
    engine = create_engine('sqlite:///' + database_filepath)
    messages_categories = pd.read_sql_table('disaster_messages', engine)
    messages = messages_categories['message']
    target_categories = messages_categories.iloc[:, 4:40]
    return messages, target_categories, target_categories.columns


def tokenize(text):
    '''Function to normalize and tokenize text messages'''
    tokens = word_tokenize(text.lower().strip())
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token))
    return clean_tokens


def build_model():
    '''Model pipeline definition initialization. Features used: Vector count, TFIDF,
    Classifier: MultiOutputClassifier
    Grid parameter Search to evaluate best parameters
    return model cv'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=10, min_samples_leaf=1, max_features='auto', n_jobs=-1)))
    ])
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.5, 0.75, 1.0),
                  'vect__max_features': (None, 5000, 10000),
                  'tfidf__use_idf': (True, False),
                  'clf__estimator__n_estimators': [50, 100, 200],
                  'clf__estimator__min_samples_split': [2, 3, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    '''Function to evaluate model performance for each category using F1 Score, Precision, Recall'''
    y_pred = model.predict(x_test)
    for i in range(36):
        print(y_test.columns[i], ':')
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], target_names=category_names))


def save_model(model, model_filepath):
    '''Function to save model as pickle file'''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    '''Main function: Split dataset in to train and test set, Initialize model,
    fit (train) the model evaluate and save the model'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        messages, target_categories, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(messages, target_categories, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(x_train, y_train)
        model = model.best_params_
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

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