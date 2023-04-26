import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.notebook import tqdm
from sklearn.metrics import  classification_report
import gzip
import re
from transformers import BertTokenizer,BertModel,BertTokenizerFast, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score
import nltk 
nltk.download('words')


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased',use_fast = True)
class TrainValDataset(Dataset):
    def __init__(self,evidences, claims,target):
        
        self.evidences = evidences
        self.claims = claims 
        self.target = target

    def __len__(self):
        return len(self.evidences)
    def __getitem__(self,index):
    
        return (self.evidences[index],self.claims[index]), self.target[index]

class BertClassifier(nn.Module):


    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768*2, 128)
        self.classifier = nn.Linear(128,3)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()


    def forward(self, input_id1, mask1,input_id2,mask2):

        with torch.no_grad():
          _,claims_bert_out = self.bert(input_ids= input_id1, attention_mask=mask1,return_dict=False)
          _,evi_bert_out = self.bert(input_ids= input_id2, attention_mask=mask2,return_dict=False)
        # print(torch.Tensor(claims_bert_out).shape)
      
        bert_out = torch.cat((claims_bert_out, evi_bert_out),dim=-1)
        
        # out = self.rest_ofLayers(bert_out)

        # _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(bert_out)
        linear_output = self.elu(self.linear(dropout_output))
        final_layer = self.classifier(linear_output)

        return final_layer

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
                mask1 = data[0]['attention_mask'].to(device)
                input_id1 = data[0]['input_ids'].squeeze(1).to(device)

                mask2 = data[1]['attention_mask'].to(device)
                input_id2 = data[1]['input_ids'].squeeze(1).to(device)

                output = model(input_id1, mask1, input_id2, mask2)
                # output = model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                curr_loss = loss.item()*target.size(0)
                train_loss+=curr_loss


                # p_bar.set_postfix(loss = curr_loss)
                p_bar.set_description(f"Epoch: {epoch+1}")
                p_bar.update()
            train_loss = train_loss/len(train_loader.dataset)

            model.eval()
            with torch.no_grad():
                for data, target in dev_loader:
                    target = target.to(device)
                    mask1 = data[0]['attention_mask'].to(device)
                    input_id1 = data[0]['input_ids'].squeeze(1).to(device)

                    mask2 = data[1]['attention_mask'].to(device)
                    input_id2 = data[1]['input_ids'].squeeze(1).to(device)

                    output = model(input_id1, mask1, input_id2, mask2)
                    # output = model(data)

                    loss = criterion(output, target)
                    
                    curr_loss = loss.item()*target.size(0)
                    val_loss+=curr_loss
            
                val_loss = val_loss/len(dev_loader.dataset)
            scheduler.step(val_loss)
        print('Epoch: {} \tValidation Loss: {:.6f}\t Train Loss: {:.6f} LR: {}'.format(epoch+1,val_loss, train_loss, optimizer.param_groups[0]['lr']))


def load_df(path, train=True):
    words = set(nltk.corpus.words.words())  
    columns = ["verifiable", "label",'claim','evidence_text']
    df = pd.read_csv(path)
    df = df[columns]
    df['evidence_text'] = df['evidence_text'].apply(lambda x: x.lower())
    df['claim'] = df['claim'].apply(lambda x: x.lower())


    df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(']','').replace('[',''))
    # df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(r'\t',' '))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: x.replace(r'[0-9] ',' '))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-lrb-',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-rrb-',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'[0-9]+\t',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'\t',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-rsb-',' ',x))
    df['evidence_text'] = df['evidence_text'].apply(lambda x: re.sub(r'-lsb',' ',x))




    return df

# tokenizer.to(device)
def preprocess(df):


    label_dict = {
        'SUPPORTS':0,
        'NOT ENOUGH INFO':1,
        'REFUTES':2
    }


  

    claim = df['claim'].apply(lambda x: tokenizer(x,padding='max_length', max_length = 512, truncation=True,return_tensors="pt")).values
    evidence = df['evidence_text'].apply(lambda x: tokenizer(x,padding='max_length', max_length = 512, truncation=True,return_tensors="pt")).values



    label = df['label'].apply(lambda x:label_dict[x]).values
    label = torch.LongTensor(label)
    return claim, evidence, label

def train_task(model, train_loader, dev_loader, n_epochs):
    # class_weights = [1,1,1] 

    # class_weights = torch.tensor(class_weights,dtype=torch.float)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, factor = 0.85, patience = 2)

    training(model, train_loader, dev_loader, criterion, optimizer, scheduler, n_epochs=n_epochs)

def evaluate_model():
    num_workers = 2
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = load_df("dev_data_final.csv", train=True)
    df = df.dropna()
    # df.sample(n=50000, weights='label', random_state=1).reset_index(drop=True)

    # df = df[:len(df)//3]
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 10000))).reset_index(drop=True)

    cl,ev, tar = preprocess(df)

    train_data = TrainValDataset(ev, cl,tar)
    train_data, val_data = torch.utils.data.random_split(train_data, [0.8,0.2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers=num_workers)




    model = BertClassifier(dropout = 0.5)
    weights = torch.load('bert_project.pt',map_location=torch.device('cpu'))

    model.load_state_dict(weights)
    model = model.to(device)

    model.eval()

    predicted,trues = [],[]
    for x,y in tqdm(dev_loader):
        # x,y = x.to(device),y.to(device)
        mask1 = x[0]['attention_mask'].to(device)
        input_id1 = x[0]['input_ids'].squeeze(1).to(device)
        mask2 = x[1]['attention_mask'].to(device)
        input_id2 = x[1]['input_ids'].squeeze(1).to(device)
        with torch.no_grad():
            output = model(input_id1, mask1, input_id2, mask2)

            _, predict = torch.max(output.data, 1)
            predicted.extend(predict.cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())

    print(classification_report(trues,predicted))

def train():
    num_workers = 2
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = load_df("train_data_final.csv", train=True)
    df = df.dropna()

    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 10000))).reset_index(drop=True)

    cl,ev, tar = preprocess(df)

    train_data = TrainValDataset(ev, cl,tar)
    train_data, val_data = torch.utils.data.random_split(train_data, [0.8,0.2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers=num_workers)

    model = BertClassifier(dropout = 0.5)
    weights = torch.load('bert_project.pt')

    model.load_state_dict(weights)
    model = model.to(device)
    train_task(model, train_loader, dev_loader, 3)

    torch.save(model.state_dict(), "bert_project.pt")


if __name__ == '__main__':

  # train()
  evaluate_model()

