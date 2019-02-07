import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



def load_data(database_filepath):
    '''
    INPUT
    database_filepath : file path of the database containing the data
    OUTPUT
    X : np.array containing the messages
    Y : np.array containing the categories related to the messages
    labels : list with the categories' names
    '''
    engine = create_engine('sqlite:///'+database_filepath+'.db').connect()
    df = pd.read_sql_table('message',con=engine)
    #remove http links
    for x in df['message']:
        urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', x)
        for u in urls:
            df[df.message==x]['message'] = x.replace(u,'')
    
    #remove data without label        
    df['sum1'] = df.sum(axis=1)
    df = df[df['sum1']>0].copy() 
    df = df.drop(['sum1'],axis=1)
    
    X = df[['message']].astype(str).values.ravel()
    Y = df.drop(['id','message','original','genre'],axis=1).astype(int)
    labels = Y.columns
    Y = Y.values
    return X,Y, labels


def tokenize(text,stopword=True,lemma=True):
    '''
    INPUT
    text : string to be tokenized
    stopword : boolean set by default to True to remove stopwords
    lemma : boolean set by default to True to lemmatize the string
    OUTPUT
    clean_tokens : list of strings
    '''
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = text.replace('*','').lower()
    words = word_tokenize(text)
    if stopword:
        words = [w.strip() for w in words if w not in stopwords.words("english")]
    
    if lemma:
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in words:
            clean_tok = lemmatizer.lemmatize(tok,pos='v')
            clean_tokens.append(clean_tok)
    else:
        clean_tokens = clean_words
    return clean_tokens
    


def build_model():
    '''
    OUTPUT
    model : pipeline model passed into GridSearch
    '''
    model = Pipeline([
    ('vect',  CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('multiout', MultiOutputClassifier(estimator = AdaBoostClassifier(random_state=42), n_jobs=-1 ))
    ])
    
    parameters = {
        'multiout__estimator__n_estimators': [100,50],
        'multiout__estimator__learning_rate': [0.1,1,10]
        
    }
    model = GridSearchCV(model,parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model : model to evaluate
    X_test : test dataset to be evaluated
    Y_test : labels associated to the test dataset
    category_names : list for string corresponding to the labels'names
    OUTPUT
    Print
    '''
    y_pred = model.predict(X_test)
    print('Scoring from the testing set')
    print(classification_report(pd.DataFrame(Y_test).values,
                               pd.DataFrame(y_pred).values,
                               target_names=category_names)) 
    return

def save_model(model, model_filepath):
    '''
    INPUT
    model : machine learning model to be saved
    model_filepath : file path
    OUTPUT
    Save the model under the filepath provided
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #database_filepath = 'DisasterMessage'
        #model_filepath = 'classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
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