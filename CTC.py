#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchnlp.nn as nlp
import Levenshtein
from ctcdecode import CTCBeamDecoder
from phoneme_list import *


# In[3]:


train = np.load('wsj0_train.npy', allow_pickle=True, encoding='latin1')


# In[4]:


train_label = np.load('wsj0_train_merged_labels.npy', allow_pickle=True)


# In[5]:


train_X = [torch.from_numpy(sequence) for sequence in train]


# In[6]:


train_X_lens = torch.LongTensor([len(seq) for seq in train_X])


# In[7]:


train_Y = [torch.LongTensor(label) for label in train_label]


# In[8]:


train_Y_lens = torch.LongTensor([len(seq) for seq in train_Y])


# In[9]:


train_X = pad_sequence(train_X)
train_Y = pad_sequence(train_Y, batch_first=True)


# In[10]:


# torch.save(train_Y, 'train_Y.pt')


# In[11]:


# train_X = torch.load('train_X.pt')


# In[12]:


# train_X_lens = torch.load('train_X_lens.pt')


# In[13]:


# train_Y = torch.load('train_Y.pt')


# In[14]:


# train_Y_lens = torch.load('train_Y_lens.pt')


# In[15]:


val = np.load('wsj0_dev.npy', allow_pickle=True, encoding='latin1')


# In[16]:


val_label = np.load('wsj0_dev_merged_labels.npy', allow_pickle=True)


# In[17]:


val_X = [torch.from_numpy(sequence) for sequence in val]
val_X_lens = torch.LongTensor([len(seq) for seq in val_X])
val_Y = [torch.LongTensor(label) for label in val_label]
val_Y_lens = torch.LongTensor([len(seq) for seq in val_Y])


# In[18]:


val_X = pad_sequence(val_X)
val_Y = pad_sequence(val_Y, batch_first=True)


# In[19]:


print('X', train_X.size(), train_X_lens)
print('Y', train_Y.size(), train_Y_lens)


# In[20]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(40, 512, num_layers=7, bidirectional=True)
#         self.bn = nn.BatchNorm1d(1024)
#         self.linear1 = nn.Linear(1024, 1024)
#         self.act = nn.ReLU()
        self.linear2 = nn.Linear(1024, 47)
#         self.dropout = nlp.LockedDropout(0.5)
    
    def forward(self, X, lengths):
        X = X.permute(1,0,2)
#         X = self.dropout(X)
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_X = packed_X.cuda()
        out = self.lstm(packed_X)[0]
        
#         out = self.lstm2(out)[0]
#         out = self.lstm3(out)[0]
        out, out_lens = pad_packed_sequence(out)
        
#         out = out.permute(0, 2, 1)
#         out = self.bn(out)
#         out = out.permute(0, 2, 1)
        # Log softmax after output layer is required for use in `nn.CTCLoss`.
#         out = self.linear1(out)
#         out = self.act(out)
        out = self.linear2(out).log_softmax(2)
        return out, out_lens


# In[21]:


class SeqDataset(Dataset):
    def __init__(self, x, y, x_len, y_len):
        self.x = x
        self.y = y
        self.x_len = x_len
        self.y_len = y_len
    def __getitem__(self, index):
        x_item = self.x[:,index,:]
        y_item = self.y[index]
        x_len = self.x_len[index]
        y_len = self.y_len[index]
        return x_item, y_item, x_len, y_len
    def __len__(self):
        return len(self.y)

# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
# def collate_lines(seq_list):
#     inputs,targets = zip(*seq_list)
#     lens = [len(seq) for seq in inputs]
#     seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
#     inputs = [inputs[i] for i in seq_order]
#     targets = [targets[i] for i in seq_order]
#     return inputs,targets


# In[22]:


train_dataset = SeqDataset(train_X, train_Y, train_X_lens, train_Y_lens)


# In[23]:


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)


# In[24]:


val_dataset = SeqDataset(val_X, val_Y, val_X_lens, val_Y_lens)


# In[25]:


val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32)


# In[ ]:


model = Model()
model = model.cuda()


# In[ ]:


criterion = nn.CTCLoss()


# In[ ]:


import time


# In[ ]:


def train(epoch, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loss = 0
    for epoch in range(epoch):
        batch_id=0
        before = time.time()
        print("\nTraining", len(train_loader), " batches")
        for X, Y, X_len, Y_len in train_loader:
            batch_id+=1
    #         X = X.cuda()
            Y = Y.cuda()
    #         X_len = X_len.cuda()
            Y_len = Y_len.cuda()
    #         X_lens = torch.LongTensor([len(seq) for seq in inputs]).to('cuda')
    #         Y_lens = torch.LongTensor([len(seq) for seq in targets]).to('cuda')
            out, out_lens = model(X, X_len)
            loss = criterion(out, Y, out_lens, Y_len)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_id % 100 == 0:
                after = time.time()
                train_loss /= 100
                print("Time elapsed:", after - before, "At batch", batch_id, "Loss:", train_loss, end='\r')
                before = after
                train_loss = 0
            torch.cuda.empty_cache()
        val_loss = 0
        batch_id=0
        for X, Y, X_len, Y_len in val_loader:
            batch_id += 1
            Y = Y.cuda()
            Y_len = Y_len.cuda()
            out, out_lens = model(X, X_len)
            loss = criterion(out, Y, out_lens, Y_len)
            val_loss += loss.item()
            torch.cuda.empty_cache()
        after = time.time()
        val_loss = val_loss / batch_id
        print("\nTime elapsed:", after - before, "Val_Loss:", val_loss, end='\r')
        torch.save(model, 'ctc_epoch_'+str(epoch)+'.pth')
        
        # distance
        out_array = []
        test_lens = torch.LongTensor()
        with torch.no_grad():
            for X, _, X_len, _ in val_loader:
            #     X = X.cuda()
            #     X_len = X_len.cuda()
                out, out_lens = model(X, X_len)
            #     print(out.shape, out_lens.shape)
            #     out = out.to('cpu')
            #     out_lens = out_lens.to('cpu')
                out_array.append(out.cpu())
                test_lens = torch.cat((test_lens, out_lens), dim=0)
                torch.cuda.empty_cache()
        decoder = CTCBeamDecoder(['$']*47, beam_width=100, num_processes=12, log_probs_input=True)
        len_idx = 0
        utterance = []
        for arr in out_array:
            probs = arr.transpose(0, 1)
            out, _, _, out_lens = decoder.decode(probs, test_lens[len_idx:len_idx+probs.shape[0]])
            len_idx += probs.shape[0]
            for i, utt in enumerate(out[:,0,:]):
                sentence = ''
                for idx in utt[:out_lens[:,0].numpy()[i]]:
                    sentence += PHONEME_MAP[idx]
                utterance.append(sentence)
        true_label = []
        for i, utt in enumerate(val_Y):
                sentence = ''
                for idx in utt[:val_Y_lens[i]]:
                    sentence += PHONEME_MAP[idx]
                true_label.append(sentence)
        dis = 0
        for i in range(len(utterance)):
            dis += Levenshtein.distance(utterance[i], true_label[i])
        print('\nDistance:', dis / len(utterance))


# In[ ]:


# torch.save(model, 'ctc.pth')


# In[ ]:


# model = torch.load('ctc.pth')


# In[41]:


train(9, 0.001)


# In[ ]:


train(5, 0.0001)


# In[27]:


test = np.load('wsj0_test.npy', allow_pickle=True, encoding='latin1')
test_X = [torch.from_numpy(sequence) for sequence in test]
test_X_lens = torch.LongTensor([len(seq) for seq in test_X])
test_X = pad_sequence(test_X)


# In[28]:


class SeqDataset(Dataset):
    def __init__(self, x, x_len):
        self.x = x
#         self.y = y
        self.x_len = x_len
#         self.y_len = y_len
    def __getitem__(self, index):
        x_item = self.x[:,index,:]
#         y_item = self.y[index]
        x_len = self.x_len[index]
#         y_len = self.y_len[index]
        return x_item, x_len
    def __len__(self):
        return self.x.shape[1]


# In[30]:


test_dataset = SeqDataset(test_X, test_X_lens)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)


# In[42]:


out_array = []
test_lens = torch.LongTensor()


# In[43]:


with torch.no_grad():
    for X, X_len in test_loader:
    #     X = X.cuda()
    #     X_len = X_len.cuda()
        out, out_lens = model(X, X_len)
    #     print(out.shape, out_lens.shape)
    #     out = out.to('cpu')
    #     out_lens = out_lens.to('cpu')
        out_array.append(out.cpu())
        test_lens = torch.cat((test_lens, out_lens), dim=0)
        torch.cuda.empty_cache()


# In[47]:


decoder = CTCBeamDecoder(['$']*47, beam_width=100, num_processes=12, log_probs_input=True)
len_idx = 0
utterance = []
for arr in out_array:
    probs = arr.transpose(0, 1)
    out, _, _, out_lens = decoder.decode(probs, test_lens[len_idx:len_idx+probs.shape[0]])
    len_idx += probs.shape[0]
    for i, utt in enumerate(out[:,0,:]):
        sentence = ''
        for idx in utt[:out_lens[:,0].numpy()[i]]:
            sentence += PHONEME_MAP[idx]
        utterance.append(sentence)


# In[48]:


import pandas as pd
output_df = pd.DataFrame()
output_df['Id'] = np.asarray(range(len(utterance)))
output_df['Predicted'] = np.asarray(utterance)


# In[49]:


output_df.to_csv('result.csv', index=None)


# In[ ]:




