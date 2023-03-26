import pandas as pd
import numpy as np
import os
import gzip

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# Credit: From PyTorch's documentation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def runProject():
    print(f"Reading Training Data Final")
    train_data = pd.read_csv('train_data_final.csv')
    print(train_data.head(5))

    print('Develop GloVe word embeddings')
    # Hyperparameters
    BATCH_SIZE = 1
    EMBEDDING_DIM = 100
    LSTM_HIDDEN_DIM = 256
    LSTM_DROPOUT = 0.33
    LINEAR_DIM = 128
    LEARNING_RATE = 0.3
    MOMENTUM = 0.9
    ELU_ALPHA = 0.5
    SCHEDULER_STEP_SIZE = 5
    SCHEDULER_GAMMA = 0.5
    NUM_EPOCHS = 20
    SPELLING_EMBEDDING_DIM = 20

    embeddings_dict = {}
    vocab = {'<PAD>', '<UNK>'}
    tag_to_ix = {'<PAD>': 0}

    with gzip.open('glove.6B.100d.gz', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    for sentence, tags in train_data:
        vocab.update(sentence)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    # for sentence, tags in dev_data:
    #     vocab.update(sentence)
    # for sentence in test_data:
    #     vocab.update(sentence)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    # ix_to_word = {v: k for k, v in word_to_ix.items()}

    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
    for word in vocab:
        index = word_to_ix[word]
        if word in embeddings_dict:
            vector = embeddings_dict[word]
        elif word.lower() in embeddings_dict:
            vector = embeddings_dict[word.lower()]
        else:
            vector = np.random.rand(EMBEDDING_DIM)
        embedding_matrix[index] = vector

    VOCAB_SIZE = len(word_to_ix)
    TAGS_SIZE = len(tag_to_ix)

    # Load data
    train_dataset = NERDataset(train_data, word_to_ix, tag_to_ix)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # dev_dataset = NERDataset(dev_data, word_to_ix, tag_to_ix)
    # dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = BLSTM_GLOVE(VOCAB_SIZE, EMBEDDING_DIM, LSTM_HIDDEN_DIM, LINEAR_DIM, TAGS_SIZE, LSTM_DROPOUT, ELU_ALPHA,
                        embedding_matrix, SPELLING_EMBEDDING_DIM, word_to_ix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    if os.path.isfile('blstm_glove.pt'):
        print('blstm_glove.pt exists. Loading existing model...')
        model = torch.load('blstm_glove.pt')
        model.to(device)
    else:
        print('blstm_glove.pt does not exist. Training a new model...')
        total_loss = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            for i, (x, y, x_original, x_spelling) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                y_pred_scores = model(x, x_spelling)
                y_pred = torch.flatten(y_pred_scores, start_dim=0, end_dim=1)
                y = torch.flatten(y)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
            print(
                f'Epoch {epoch + 1} / {NUM_EPOCHS}, training loss: {np.average(total_loss):.5f}, '
                f'learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
            total_loss = []
            scheduler.step()
            # if epoch == 0 or (epoch + 1) % 5 == 0:
            #     predict2(model, dev_loader, f'Epoch {epoch + 1} / {NUM_EPOCHS}', tag_to_ix)
        predict2(model, train_loader, f'Epoch {epoch + 1} / {NUM_EPOCHS}', tag_to_ix)
        torch.save(model, 'blstm_glove.pt')

    # Prediction for all cases (dev, test, and dev for perl)
    print('Beginning Predictions!')
    # predict_perl2(model, dev_loader, 'prediction2.txt', ix_to_tag)
    # predict_dev2(model, dev_loader, 'dev2.out', ix_to_tag)
    predict_train(model, train_data, 'test2.out', word_to_ix, ix_to_tag)
    print('Predictions Done!')
    #
    # with open('word_to_ix_2.pkl', 'wb') as f:
    #     pickle.dump(word_to_ix, f)
    #
    # with open('tag_to_ix_2.pkl', 'wb') as f:
    #     pickle.dump(tag_to_ix, f)
    return


def prepare_sequence(seq, to_ix, use_unk=False):
    if use_unk:
        indices = [to_ix[w] if w in to_ix else to_ix['<UNK>'] for w in seq]
    else:
        indices = [to_ix[w] for w in seq]
    return indices


def get_spelling_feature(sentence):
    result = []
    for word in sentence:
        # PAD = 0
        if word == '<PAD>':
            result.append(0)
        # ALL LOWER = 1
        elif word.islower():
            result.append(1)
        # ALL UPPER = 2
        elif word.isupper():
            result.append(2)
        # FIRST UPPER = 3
        elif word[0].isupper():
            result.append(3)
        # OTHERS = 4
        else:
            result.append(4)
    return result

# Used to predict on a development data loader
# Writes statistics to console
def predict2(model, data_loader, message, tag_to_ix):
    all_y = []
    all_y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y, x_original, x_spelling in data_loader:
            x, y = x.to(device), y.to(device)

            y_pred_scores = model(x, x_spelling)
            y_pred = torch.argmax(y_pred_scores, dim=2)
            y_pred_flat = torch.flatten(y_pred).tolist()
            y_flat = torch.flatten(y).tolist()

            for i in range(len(y_pred_flat)):
                if y_flat[i] == tag_to_ix['<PAD>']:
                    break
                all_y.append(y_flat[i])
                all_y_pred.append(y_pred_flat[i])

    print(message, classification_report(all_y, all_y_pred))


# Used to predict on a test data, list of sentences
# Writes the output to a file, i.e. to test.out
def predict_train(model, sentences, fname, word_to_ix, ix_to_tag):
    outputs = []
    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            spelling_sentence = [get_spelling_feature(sentence)]
            spelling_sentence = torch.from_numpy(np.array(spelling_sentence, dtype=np.int64)).to(device)

            transformed_sentence = [prepare_sequence(sentence, word_to_ix, use_unk=True)]
            transformed_sentence = torch.from_numpy(np.array(transformed_sentence, dtype=np.int64)).to(device)

            y_pred_scores = model(transformed_sentence, spelling_sentence)
            y_pred = torch.argmax(y_pred_scores, dim=2)
            y_pred_flat = torch.flatten(y_pred).tolist()

            idx = 1
            output = []
            for i in range(len(y_pred_flat)):
                word = sentence[i]
                pred = ix_to_tag[y_pred_flat[i]]
                if word == '<PAD>':
                    break
                output.append((idx, word, pred))
                idx += 1
            outputs.append(output)

    with open(fname, 'w', newline='\n') as f:
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                idx, word, pred = outputs[i][j]
                f.write(f'{idx} {word} {pred}\n')
            if i != len(outputs) - 1:
                f.write('\n')



class NERDataset(Dataset):
    def __init__(self, data, word_to_ix, tag_to_ix):
        # Retrieves longest sentence, for padding
        max_sentence_len = max([len(sentence) for sentence, tags in data])
        self.x = []
        self.x_original = []
        self.y = []
        self.x_spelling = []

        for sentence, tags in data:
            # Pad the sentences to the same length
            padded_sentence = sentence.copy()
            padded_tags = tags.copy()
            while len(padded_sentence) < max_sentence_len:
                padded_sentence.append('<PAD>')
                padded_tags.append('<PAD>')
            # Convert to indices
            transformed_sentence = prepare_sequence(padded_sentence, word_to_ix, use_unk=True)
            transformed_tags = prepare_sequence(padded_tags, tag_to_ix)
            # Get spelling indices
            spelling_sentence = get_spelling_feature(padded_sentence)
            # Add to dataset
            self.x.append(transformed_sentence)
            self.x_original.append(padded_sentence)
            self.y.append(transformed_tags)
            self.x_spelling.append(spelling_sentence)

        self.x = torch.from_numpy(np.array(self.x, dtype=np.int64)).to(device)
        self.y = torch.from_numpy(np.array(self.y, dtype=np.int64)).to(device)
        self.x_spelling = torch.from_numpy(np.array(self.x_spelling, dtype=np.int64)).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_original[index], self.x_spelling[index]


# Bidirectional LSTM Model with GloVe embeddings
class BLSTM_GLOVE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, linear_dim, tags_size, lstm_dropout, elu_alpha,
                 embeddings, spelling_embedding_dim, word_to_ix):
        super(BLSTM_GLOVE, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embeddings_word = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False,
                                                            padding_idx=word_to_ix['<PAD>'])
        self.embeddings_spelling = nn.Embedding(num_embeddings=5, embedding_dim=spelling_embedding_dim,
                                                padding_idx=0)
        self.dropout_pre_lstm = nn.Dropout(lstm_dropout)
        self.lstm = nn.LSTM(embedding_dim + spelling_embedding_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        self.dropout_post_lstm = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU(alpha=elu_alpha)
        self.linear2 = nn.Linear(linear_dim, tags_size)

    def forward(self, x_word, x_spelling):
        x1 = self.embeddings_word(x_word)
        x2 = self.embeddings_spelling(x_spelling)
        x = torch.cat((x1, x2), dim=2).to(device)
        x = self.dropout_pre_lstm(x)
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout_post_lstm(out)
        out = self.linear(out)
        out = self.elu(out)
        out = self.linear2(out)
        return out


if __name__ == "__main__":
    runProject()
