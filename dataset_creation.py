
import pandas as pd
import json
from os import listdir
from os.path import isfile, join
from numpy import int64
import ast

onlyfiles = [f for f in listdir('wiki-pages/') if isfile(join('wiki-pages/', f))]

evidences = dict()
evidences['id'] = list()
evidences['lines'] = list()
for f in onlyfiles:
    file  = 'wiki-pages/'+f
    with open(file,'r') as f_:
        
        for l in f_:
            l_ = json.loads(l)
            data = json.loads(l.rstrip("\n"))
            evidences['id'].append(data['id'])
            evidences['lines'].append(data["lines"].split('\n'))
            

df_evidences = pd.DataFrame.from_records(evidences)

#df_evidences.to_csv('evidences_from_wikipages.csv')


train_data = pd.read_json('/content/drive/MyDrive/train.jsonl',lines = True)
#df_evidences = pd.read_csv('/content/drive/MyDrive/evidences_from_wikipages.csv')



print('Training Samples: ',len(train_data))
print('Maximum length of the claim in the train data: ',max(len(d) for d in train_data.claim.values))
print('Minimum length of the claim in the train data: ',min(len(d) for d in train_data.claim.values))
print('Maximum length of the evidence in the train data: ',max(len(d) for d in train_data.evidence.values))
print('Minimum length of the evidence in the train data: ',min(len(d) for d in train_data.evidence.values))


train_evidence_id = train_data['evidence'].apply(lambda x: [k[2] for k in x[0]])
train_evidence_sentence_id = train_data['evidence'].apply(lambda x: [k[3] for k in x[0]])

train_data['evidence_id'] = train_evidence_id
train_data['evidence_sentence_id'] = train_evidence_sentence_id

train_data = train_data.explode(['evidence_id','evidence_sentence_id'])
merged_train = pd.merge(train_data, df_evidences, how='left',left_on = 'evidence_id',right_on = 'id')

merged_train.verifiable.value_counts()
merged_train.drop('id_y',axis =1,inplace = True)
merged_train.drop('Unnamed: 0',axis =1,inplace = True)

merged_train = merged_train.fillna({'claim':'No claim', 'evidence': 'No evidence', 'evidence_id':'nil','evidence_sentence_id':-1,'lines':''})

lines = merged_train['lines'].values.tolist()
sentence_ids = merged_train['evidence_sentence_id'].values.tolist()
evidence_text = []

for l,s in zip(lines,sentence_ids):
  if not l:
    evidence_text.append('')
  else:
    l_ = ast.literal_eval(l)
    if s!=-1:
      evidence_text.append(l_[int(s)]) 
    else:
      evidence_text.append('')

merged_train['evidence_text'] = evidence_text
merged_train = merged_train.groupby('id_x').agg(list)

final_train = merged_train.copy()

final_train['verifiable'] = final_train['verifiable'].apply(lambda x: x[0])
final_train['claim'] = final_train['claim'].apply(lambda x: x[0])
final_train['evidence'] = final_train['evidence'].apply(lambda x: x[0])
final_train['evidence_id'] = final_train['evidence_id']#.apply(lambda x: x[0])
final_train['evidence_sentence_id'] = final_train['evidence_sentence_id']#.apply(lambda x: x[0] if x[0]!=None else x[0])
final_train['lines'] = final_train['lines'].apply(lambda x: x[0])
final_train['label'] = final_train['label'].apply(lambda x: x[0])


final_train.to_csv('train_data_final.csv')
