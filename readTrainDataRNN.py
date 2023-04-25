import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from keras.utils import pad_sequences


def runProject():
    print(f"Reading Training Data Final")

    # Load the dataset
    df = pd.read_csv('train_data_final.csv')

    # Separate the claims and evidences into separate arrays
    claims = np.array(df['claim'])
    evidences = np.array(df['evidence_text'])

    # Vectorize the claims and evidences using TF-IDF
    vectorizer = TfidfVectorizer()
    X_claims = vectorizer.fit_transform(claims)
    X_evidences = vectorizer.transform(evidences)

    # Concatenate the claims and evidences vectors
    X = np.concatenate((X_claims.toarray(), X_evidences.toarray()), axis=1)

    # Convert the labels to numeric format
    y = df['label'].replace({'SUPPORTS': 1, 'REFUTES': 0, 'NOT ENOUGH INFO': 2})

    # Pad the input sequences to have the same length
    X = pad_sequences(X, maxlen=500, padding='post')

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the RNN model architecture
    model = Sequential()
    model.add(Embedding(input_dim=X.shape[1], output_dim=128, input_length=X.shape[1]))
    model.add(SimpleRNN(units=64))
    model.add(Dense(units=3, activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[early_stopping])

    # Evaluate the model on the test data and print the classification report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    runProject()
