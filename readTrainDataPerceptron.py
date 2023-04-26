import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


def runProject():
    print(f"Reading Training Data Final")

    # Load the dataset
    df = pd.read_csv('train_data_final.csv')

    # Separate the claims and evidences into separate arrays
    # claims = np.array(df['claim'])
    # # df['evidence_text_new'] = df['evidence_text'].apply(lambda x: process_evidences(x))
    # evidences = np.array(df['evidence_text'])
    # # evidences = np.array(df['evidence_text_new'])
    #
    # # Vectorize the claims and evidences using TF-IDF
    # vectorizer = TfidfVectorizer()
    # X_claims = vectorizer.fit_transform(claims)
    # X_evidences = vectorizer.fit_transform(evidences)
    #
    # # Concatenate the claims and evidences vectors
    # X = hstack([X_claims, X_evidences])
    #
    # # Create the target variable
    # y = np.array(df['label'])
    # y = df['label'].replace({'SUPPORTS': 1, 'REFUTES': 0, 'NOT ENOUGH INFO': 2})

    # Separate the claims and evidences into separate arrays
    claims = df['claim']
    evidences = df['evidence_text']

    # Concatenate the claims and evidences into a single text string
    texts = claims + " " + evidences

    # Convert the labels to numeric format
    # y = np.array(df['label'])
    y = df['label'].replace({'SUPPORTS': 1, 'REFUTES': 0, 'NOT ENOUGH INFO': 2})

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the perceptron model and fit the training data
    clf = Perceptron(max_iter=1000, tol=1e-3)
    clf.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


# def process_evidences(evidence):
#     import re
#     text = evidence
#     text = text.replace("[", "")
#     text = text.replace("]", "")
#     text = text.replace("\t", "")
#     return text


if __name__ == "__main__":
    runProject()
