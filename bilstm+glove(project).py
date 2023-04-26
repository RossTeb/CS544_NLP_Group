
import pandas as pd
import torch
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.notebook import tqdm
from sklearn.metrics import  classification_report
import gzip
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
nltk.download('words')

class TrainValDataset(Dataset):
    def __init__(self,evidences, claims,target):
        self.evidences = evidences
        self.claims = claims

        
        self.target = target

    def __len__(self):
        return len(self.evidences)
    def __getitem__(self,index):
    
        return (self.evidences[index],self.claims[index]), self.target[index]

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_matrix, n_classes, emb_dim, hidden_dim, output_dim, num_layers, dropout, pad_idx):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        

        self.embedding = nn.Embedding(vocab_size,emb_dim, padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb_matrix.astype('float32')))

        self.lstm = nn.LSTM(200, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(output_dim,n_classes)
        
        self.fc1 = nn.Linear(output_dim,64)

        self.fc2 = nn.Linear(64, n_classes)
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        device = next(self.parameters()).device

        evidences,claims = x
        evidences = evidences.to(device)
        claims = claims.to(device)


        ## get them to equal length
        ev_len = evidences.shape[1]
        cl_len = claims.shape[1]
        if ev_len > cl_len:
            claims = torch.cat((claims,torch.zeros((claims.shape[0],ev_len-cl_len),dtype=torch.long).to(device)),dim=-1)
        elif cl_len > ev_len:
            evidences = torch.cat((evidences,torch.zeros((evidences.shape[0],cl_len-ev_len),dtype=torch.long).to(device)),dim=-1)



        ev_emb = self.embedding(evidences)
        cl_emb = self.embedding(claims)
        inp = torch.cat((ev_emb,cl_emb),dim=-1)

        #### concatenate first

        # print(ev_len,cl_len)

        # ev_cl = torch.cat((evidences,claims),dim=-1)
        # print('HERE')
        # inp = self.embedding(ev_cl) 

        # print(inp.shape)

        output, (hidden, cell) = self.lstm(inp)
        lstm_output = output[:,-1,:]

        output = self.elu(self.fc(lstm_output))

        output = self.dropout_layer(output)
        output = self.classifier(output)
        # output = nn.functional.log_softmax(output, dim=-1)
        
       
        
    
        return output

import torch.nn.functional as F
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_matrix, n_classes, emb_dim, hidden_dim, output_dim, num_layers, dropout, pad_idx):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        

        self.embedding = nn.Embedding(vocab_size,emb_dim, padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb_matrix.astype('float32')))

        self.lstm = nn.LSTM(200, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(2*hidden_dim,n_classes)
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
    def forward(self, x):
        device = next(self.parameters()).device

        evidences,claims = x
        evidences = evidences.to(device)
        claims = claims.to(device)


        ## get them to equal length using pre padding
        ev_len = evidences.shape[1]
        cl_len = claims.shape[1]
        max_len = max(ev_len,cl_len)
        # print(claims.shape, evidences.shape)

        if ev_len < max_len:
            evidences = F.pad(evidences,(0,max_len-ev_len))
        if cl_len < max_len:
            claims = F.pad(claims,(0,max_len-cl_len))

        # print(claims.shape, evidences.shape)
        ## get the embeddings
        ev_emb = self.embedding(evidences)
        cl_emb = self.embedding(claims)

        ## concat the embeddings
        ev_cl_emb = torch.cat((ev_emb,cl_emb),dim=-1)
        # print(ev_cl_emb.shape)
        ## pass through the lstm
        output, (hidden, cell) = self.lstm(ev_cl_emb)

        ## get the last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = self.dropout_layer(hidden)

        ## pass through the classifier
        output = self.classifier(hidden)





       
        
    
        return output

def training(model, train_loader, dev_loader, criterion, optimizer, scheduler, n_epochs):
    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0
        device = next(model.parameters()).device

        with tqdm(total = len(train_loader)) as p_bar:
            model.train()

            for data, target in train_loader:
                target = target.to(device)
                # print(target.shape)
                
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                curr_loss = loss.item()*target.size(0)
                train_loss+=curr_loss


                p_bar.set_postfix(loss = curr_loss)
                p_bar.set_description(f"Epoch: {epoch+1}")
                p_bar.update()
            train_loss = train_loss/len(train_loader.dataset)

            model.eval()
            with torch.no_grad():
                for data, target in dev_loader:
                    target = target.to(device)
                    output = model(data)

                    loss = criterion(output, target)
                    
                    curr_loss = loss.item()*target.size(0)
                    val_loss+=curr_loss
            
                val_loss = val_loss/len(dev_loader.dataset)
            scheduler.step(val_loss)
        print('Epoch: {} \tValidation Loss: {:.6f}\t Train Loss: {:.6f} LR: {}'.format(epoch+1,val_loss, train_loss, optimizer.param_groups[0]['lr']))

def predict(model, data):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for d in tqdm(data):
            predictions = model(d)
            _, predict = torch.max(predictions.data, 1)
            all_predictions.append(predict.to(torch.device("cpu")).numpy()[0].tolist())
    
    return all_predictions


def load_df(path, train=True):
    # words = set(nltk.corpus.words.words())  
    columns = ["verifiable", "label",'claim','evidence_text']
    df = pd.read_csv(path)
    df = df[columns]

    # df['evidence_text'] = df['evidence_text'].apply(lambda x: ' '.join(w for w in x.split() if w.lower() in words or not w.isalpha()))
    # df['claim'] = df['claim'].apply(lambda x: ' '.join(w for w in x.split() if w.lower() in words or not w.isalpha()))

    df = df.replace({'evidence_text': r'-lrb-'}, {'A': ''}, regex=True)





    # df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(']','').replace('[',''))
    # df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(r'\t',' '))
    # # df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(r'[0-9] ',' '))

    df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(']','').replace('[',''))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(r'\t',' '))
    # df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(r'[0-9] ',' '))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-lrb-',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-rrb-',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'[0-9]+\t',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'\t',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-rsb-',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-lsb',' ',x))
  




    # df['evidence_text'] = df['evidence_text'].apply(lambda x: x[:min(len(x),250)])
    # df['claim'] = df['claim'].apply(lambda x: x[:min(len(x),250)])




    df['evidence_text'] = df['evidence_text'].apply(lambda x: x.lower())
    df['claim'] = df['claim'].apply(lambda x: x.lower())
    return df

def preprocess(df, vocab, stop_words):


    label_dict = {
        'SUPPORTS':0,
        'NOT ENOUGH INFO':1,
        'REFUTES':2
    }

    label = df['label'].apply(lambda x:label_dict[x]).values
    claim = df["claim"].apply(lambda x: torch.LongTensor([vocab.get(word, vocab["<UNK>"]) for word in process_sentence(x, stop_words)])).values
    evidence = df['evidence_text'].apply(lambda x: torch.LongTensor([vocab.get(word, vocab["<UNK>"]) for word in process_sentence(x, stop_words)])).values

    claim = nn.utils.rnn.pad_sequence(claim, batch_first=True, padding_value=0)
    evidence = nn.utils.rnn.pad_sequence(evidence, batch_first=True, padding_value=0)
    label = torch.LongTensor(label)
    return claim, evidence, label

def get_embedding_matrix(path, vocab):

    with gzip.open(path,'rb') as f:
        glove_emb = f.readlines()

    emb = dict()

    for e in glove_emb:
        e = e.decode()
        e = e.rstrip('\n')
        e = e.split(' ')
        w,g = e[0], list(map(float,e[1:]))
        emb[w] = g

    reverse_vocab = {v:k for k, v in vocab.items()}
    emb_matrix = [np.zeros(100)]
    for i in range(1, len(reverse_vocab)+1):
        ## generate a random embedding for every unknown word
        unk_emb = np.random.randn(100)*0.5
        emb_matrix.append(emb.get(reverse_vocab[i].lower(), unk_emb))

    emb_matrix = np.array(emb_matrix)
    return emb_matrix

    # return emb

def train_task(model, train_loader, dev_loader, n_epochs):
    # class_weights = [1,2,2] 

    # class_weights = torch.tensor(class_weights,dtype=torch.float)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scheduler = ReduceLROnPlateau(optimizer, factor = 0.85, patience = 2)

    training(model, train_loader, dev_loader, criterion, optimizer, scheduler, n_epochs=n_epochs)



def process_sentence(sentence, stop_words):
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def get_vocab(df, stop_words):
    all_text = " ".join(df[["claim", "evidence_text"]].values.flatten().tolist())
    tokens = process_sentence(all_text, stop_words)
    word_counts = Counter(tokens)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    threshold = 5
    filtered_vocab = [word for word in sorted_vocab if word_counts[word] > threshold]
    vocab_dictionary = {word: i for i, word in enumerate(filtered_vocab, 2)}
    vocab_dictionary["<UNK>"] = 1
    return vocab_dictionary

def evaluate(vocab,stopwords,emb):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = load_df("dev_data_final.csv", train=True)
    df = df.dropna()
  



    ev, cl, tar = preprocess(df,vocab, stop_words)


    val_data = TrainValDataset(ev, cl, tar)
    dev_loader = torch.utils.data.DataLoader(val_data, batch_size=32,num_workers=2)



    model = BiLSTM(vocab_size=len(emb),emb_matrix=emb, n_classes=3,
                            emb_dim=100,
                             hidden_dim=256, output_dim=128, num_layers=1, dropout=0.4, pad_idx=0)
    weights = torch.load('blstm_project.pt',map_location=torch.device('cpu'))

    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    predicted,trues = [],[]
    for x,y in tqdm(dev_loader):
        # x,y = x.to(device),y.to(device)
        with torch.no_grad():
            outputs = model(x)
            _, predict = torch.max(outputs.data, 1)
            predicted.extend(predict.cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())

    print(classification_report(trues,predicted))

def train(df,stop_words,vocab,emb):




    ev, cl, tar = preprocess(df,vocab, stop_words)




    train_data = TrainValDataset(ev, cl, tar)
    train_data, val_data = torch.utils.data.random_split(train_data, [0.8, 0.2])
    # train_data, val_data = torch.utils.data.random_split(train_data, [3, 3])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers=num_workers)



    model = BiLSTM(vocab_size=len(emb),emb_matrix=emb, n_classes=3,
                            emb_dim=100,
                             hidden_dim=256, output_dim=128, num_layers=1, dropout=0.4, pad_idx=0)
    model = model.to(device)

    train_task(model, train_loader, dev_loader, n_epochs=15)

    torch.save(model.state_dict(), "blstm_project.pt")


if __name__ == '__main__':
    num_workers = 2
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = load_df("train_data_final.csv", train=True)
    df = df.dropna()
    stop_words = string.punctuation
    vocab = get_vocab(df, stop_words)

    emb = get_embedding_matrix('glove.6B.100d.gz', vocab)
    # train(df,stop_words,vocab,emb)
    evaluate(vocab,stop_words,emb)

