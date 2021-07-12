#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import csv
from sklearn.metrics import f1_score


# In[2]:


data = pd.read_csv("./Telugu-NER/dataset/dataset-generic.csv")
#data = pd.read_csv("./medical-dataset.csv")
#data = pd.read_csv("./Telugu-NER/dataset/dataset-combined.csv")
print(data.head(10))


# In[3]:


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["{}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[4]:


getter = SentenceGetter(data)


# In[5]:


sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
sentences[0]


# In[6]:


labels = [[s[1] for s in sentence] for sentence in getter.sentences]
print(labels[0])


# In[7]:


tag_values = list(set(data["tag"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}


# In[8]:


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import RobertaModel, RobertaTokenizer
#from bertviz import head_view

torch.__version__


# In[9]:


MAX_LEN = 75
bs = 64


# In[10]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[11]:


print(torch.cuda.get_device_name(0))


# In[12]:


#model = RobertaModel.from_pretrained('subbareddyiiit/TeRobeRta', output_attentions=True)
#tokenizer = RobertaTokenizer.from_pretrained('subbareddyiiit/TeRobeRta')

from transformers import ElectraModel,ElectraConfig, ElectraTokenizer, ElectraForMaskedLM
#from bertviz import head_view

config = ElectraConfig.from_pretrained("subbareddyiiit/TeElectra")
tokenizer = ElectraTokenizer.from_pretrained("subbareddyiiit/TeElectra",output_attentions=True)
model = ElectraModel.from_pretrained("subbareddyiiit/TeElectra",config=config)

# In[13]:


from wxconv import WXC
con = WXC(order='utf2wx',lang='tel') 


# In[14]:


# tokenize wala step ## wala step jaise gunships ##shipa
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(con.convert(word))
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


# In[15]:


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]


# In[16]:


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


# In[17]:


#cut and pad to the desied length 75 bcz ab no of token increase ho gya
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")


# In[18]:


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")


# In[19]:


#attenation mask to ignore PAD token
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


# In[20]:


#10per train and validATE
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.3)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.3)


# In[21]:


# convert to torch tenors
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[22]:


#training time shuffling of the data and testing time we pass them sequentially
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


# In[23]:


import transformers
from transformers import BertForTokenClassification, AdamW, RobertaForTokenClassification, ElectraForTokenClassification

transformers.__version__


# In[24]:


model = ElectraForTokenClassification.from_pretrained(
    "subbareddyiiit/TeElectra",
    num_labels=len(tag2idx),
    output_attentions = True,
    output_hidden_states = True
)


# In[25]:


model.cuda();


# In[26]:


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


# In[1]:


from pytorchtools import EarlyStopping


# In[27]:


#schduler to reduce learning rate linearly throughout the epochs
from transformers import get_linear_schedule_with_warmup

epochs = 30
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# In[28]:


from seqeval.metrics import f1_score
from tqdm import tqdm, trange

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def lists_lists(preds, labels):
    pred_tags = []
    true_tags = []    
    for p, l in zip(preds, labels):
        temp = []
        for p_i, l_i in zip(p, l): 
            if tag_values[l_i] != "PAD":
                temp.append(tag_values[p_i])
        pred_tags.append(temp)
    for l in labels:
        temp = []
        for l_i in l:
            if tag_values[l_i] != "PAD":
                temp.append(tag_values[l_i])
        true_tags.append(temp)
    return pred_tags, true_tags
# In[2]:


## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []
early_stopping = EarlyStopping(patience=50, verbose=True)
for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    #pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
    #                             for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    #valid_tags = [tag_values[l_i] for l in true_labels
    #                              for l_i in l if tag_values[l_i] != "PAD"]
    pred_tags,valid_tags = lists_lists(predictions, true_labels)
    val_f1score = f1_score(pred_tags, valid_tags,average='macro')
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags,average='macro')))
    early_stopping(val_f1score, model)
    if early_stopping.early_stop:
        print("Early Stopping")
        break
    print()

model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# In[31]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
"""
# Plot the learning curve.
plt.plot(loss_values, 'b-o', label="training loss")
plt.plot(validation_loss_values, 'r-o', label="validation loss")

# Label the plot.
plt.title("Learning curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
"""
eval_loss = 0
eval_accuracy = 0
predictions , true_labels = [], []
# In[34]:
i =0
for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        #print(logits)
        #print(label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
        #print(predictions)
        #print(true_labels)
        

#print(predictions)
#print(true_labels)
pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]

from sklearn.metrics import classification_report
print(classification_report(pred_tags,valid_tags))
pred_tags,valid_tags = lists_lists(predictions, true_labels)
#print(pred_tags)
#print(valid_tags)

import numpy as np
from sklearn.metrics import classification_report

#print(classification_report(pred_tags,valid_tags))

from seqeval.metrics import classification_report as classification_report_seqeval

print(classification_report_seqeval(pred_tags, valid_tags))
# Plot training & validation loss values
plt.subplot(122)
tokenized_sentence = tokenizer.encode(tokenized_texts[0])
input_ids = torch.tensor([tokenized_sentence]).cuda()


# In[33]:


with torch.no_grad():
    output = model(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)


# In[35]:


# join bpe split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)


# In[56]:


for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))


# In[57]:


# In[ ]:




