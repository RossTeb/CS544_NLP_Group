import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter


# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# load the dataset
data = pd.read_csv('train_data_rnn.csv')

# split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# vectorize the claims using TF-IDF
print("Vectorizing claims...")
tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(train_data['claim'])
test_tfidf = tfidf_vectorizer.transform(test_data['claim'])

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

# numericalize the evidence sequences
print("Numericalizing evidence sequences...")
def numericalize_tokens(tokens):
    return [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]

train_sequences = [numericalize_tokens(tokens) for tokens in train_data_tokens]
test_sequences = [numericalize_tokens(tokens) for tokens in test_data_tokens]

# pad the sequences to a fixed length
print("Padding sequences...")
train_padded = nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in train_sequences], batch_first=True, padding_value=vocab['<pad>'])
test_padded = nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True, padding_value=vocab['<pad>'])

# define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, lstm_layers, dropout):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

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
        self.fc1 = nn.Linear(train_tfidf.shape[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + lstm_layers * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x2 = self.rnn.embedding(x2)
        x2, _ = self.rnn.lstm(x2)
        x1 = self.fc1(x1)
        x = torch.cat((x1, x2[:, -1, :]), dim=1)
        x = self.fc2(x)
        x = self.dropout(x)
        return torch.sigmoid(x)


# convert the data to PyTorch tensors
label_map = {'SUPPORTS': 1, 'REFUTES': 0}
train_data['label'] = train_data['label'].map(label_map)
test_data['label'] = test_data['label'].map(label_map)

train_labels = torch.from_numpy(train_data['label'].values.astype(np.float32)).unsqueeze(1)
train_tfidf = torch.from_numpy(train_tfidf.toarray().astype(np.float32))
train_padded = torch.from_numpy(train_padded.detach().numpy())


test_labels = torch.from_numpy(test_data['label'].values.astype(np.float32)).unsqueeze(1)
test_tfidf = torch.from_numpy(test_tfidf.toarray().astype(np.float32))
# test_padded = torch.from_numpy(test_padded.numpy().astype(np.long))
test_padded = torch.from_numpy(test_padded.detach().numpy())

# create datasets and dataloaders
train_dataset = TensorDataset(train_tfidf, train_padded, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64)
test_dataset = TensorDataset(test_tfidf, test_padded, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64)

# initialize the model, loss function, and optimizer
hybrid_model = HybridModel(len(vocab), 1, 128, 50, 1, 0.2)

# hybrid_model.load_state_dict(torch.load('hybrid_model.pt'))
criterion = nn.BCELoss()
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)

num_epochs = 10
# train the model
# calculate total number of batches
total_batches = len(train_loader)
print("Training the model...")
# training loop
for epoch in range(num_epochs):
    for i, (inputs1, inputs2, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = hybrid_model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{total_batches}, Loss: {loss.item():.4f}")

    # print progress at end of epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training finished.")

    hybrid_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs1, inputs2, labels = batch
            outputs = hybrid_model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs1.size(0)
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(batch)}], Loss: {loss.item():.4f}')

    test_loss /= len(test_loader.dataset)
    print("Epoch: {} Test Loss: {:.4f}".format(epoch+1, test_loss))
    # Save the model
    torch.save(hybrid_model, 'hybrid_model.pt')
    torch.save(hybrid_model.state_dict(), 'hybrid_model_state.pt')

