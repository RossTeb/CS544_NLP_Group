import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def runProject():
    print(f"Reading Training Data Final")

    # Load the dataset
    df = pd.read_csv('train_data_final.csv')

    # Separate the claims and evidences into separate arrays
    claims = df['claim']
    evidences = df['evidence_text']

    # Concatenate the claims and evidences into a single text string
    texts = claims + " " + evidences

    # Convert the labels to numeric format
    y = df['label'].replace({'SUPPORTS': 1, 'REFUTES': 0, 'NOT ENOUGH INFO': 2})

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Naive Bayes model and fit the data
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


if __name__ == "__main__":
    runProject()
