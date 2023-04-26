import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

# check if CUDA is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Using device:', device)


# load the dataset
data = pd.read_csv('train_data_final.csv')

# split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# vectorize the claims using TF-IDF
print("Vectorizing claims...")
tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(train_data['claim'])
test_tfidf = tfidf_vectorizer.transform(test_data['claim'])


train_tfidf = torch.from_numpy(train_tfidf.toarray().astype(np.float32))
train_tfidf = train_tfidf.to(device)
test_tfidf = torch.from_numpy(test_tfidf.toarray().astype(np.float32))
test_tfidf = test_tfidf.to(device)
# tokenize the evidence and claims
print("Tokenizing evidence and claims...")
tokenizer = get_tokenizer('basic_english')
train_data_tokens = list(map(tokenizer, train_data['evidence']))
test_data_tokens = list(map(tokenizer, test_data['evidence']))


# build the vocabulary
print("Building vocabulary...")
counter = Counter()
for tokens in train_data_tokens:
    counter.update(tokens)
vocab = build_vocab_from_iterator([counter.keys()], specials=['<unk>', '<pad>'])


# define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, lstm_layers, dropout):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = self.dropout(x)
        return torch.sigmoid(x)


# define the hybrid model
class HybridModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, lstm_layers, dropout):
        super(HybridModel, self).__init__()
        self.rnn = RNNModel(input_dim, output_dim, hidden_dim, embedding_dim, lstm_layers, dropout)
        self.fc1 = nn.Linear(len(vocab), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + lstm_layers * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(self, x1, x2):
        x2 = self.rnn.embedding(x2)
        x2, _ = self.rnn.lstm(x2)
        x1 = self.fc1(x1)
        x = torch.cat((x1, x2[:, -1, :]), dim=1)
        x = self.fc2(x)
        x = self.dropout(x)
        return torch.sigmoid(x)


# hybrid_model = HybridModel(len(vocab), 1, 128, 50, 1, 0.2)
# hybrid_model.to(device)
# hybrid_model.load_state_dict(torch.load('hybrid_model_state_1.pt'))
# hybrid_model.eval()

hybrid_model = torch.load("hybrid_model_stable.pt")
hybrid_model.to(device)
hybrid_model.eval()

def predict_claim(claim, model, tokenizer, vectorizer, vocab):
    # prompt the user to enter a claim

    # claim = 'the sky is blue'
    # vectorize the claim using TF-IDF
    claim_tfidf = vectorizer.transform([claim])
    claim_tfidf = torch.from_numpy(claim_tfidf.toarray().astype(np.float32))
    claim_tfidf = claim_tfidf.to(device)

    # tokenize the evidence and numericalize the sequence
    claim_tokens = tokenizer(claim)
    claim_seq = [vocab[token] if token in vocab else vocab['<unk>'] for token in claim_tokens]
    # pad the sequence to a fixed length
    claim_padded = nn.utils.rnn.pad_sequence([torch.tensor(claim_seq).to(device)], batch_first=True, padding_value=vocab['<pad>'])
    # query the model for a prediction
    with torch.no_grad():
        # logits = model(torch.tensor(claim_tfidf.toarray()).to(device), claim_padded)
        logits = model(claim_tfidf, claim_padded)
        prediction = torch.sigmoid(logits).numpy()[0]
        if prediction > 0.6:
            label = 'SUPPORTED'
        elif prediction < 0.4:
            label = 'REFUTED'
        else:
            label = 'UNLIKELY TRUE'
        # label = 'SUPPORTS' if prediction > 0.5 else 'REFUTES'
        print(f'--------------------------------------------------------------\n'
              f'The claim is {label} with a probability of {prediction * 100}'
              f'\n--------------------------------------------------------------\n')


print("--------------------------------------------------------------")
print("|             Welcome to the Magic 8 Ball                    |")
print("--------------------------------------------------------------")

while True:

    claim = input('... 1 to Quit ...\nEnter a claim to fact-check:   ')
    if claim == "1":
        break
    else:
        predict_claim(claim, hybrid_model, tokenizer, tfidf_vectorizer, vocab)


print("Have a nice day!")

