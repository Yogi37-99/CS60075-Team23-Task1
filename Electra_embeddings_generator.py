import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.autograd import Variable
import csv
import json

from transformers import ElectraTokenizer, ElectraModel, ElectraConfig
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraModel.from_pretrained('google/electra-small-discriminator', output_hidden_states = True)
model.eval()

def addpd(df, name):
    df_temp = pd.read_csv(name,delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
    df = df.append(df_temp)
    return df

def getstates(torch,model,indexed_tokens, segments_ids):
    with torch.no_grad():
        outputs = model(torch.tensor([indexed_tokens]), torch.tensor([segments_ids]))
        return outputs[0]


def computeLPSArray(pat, M, lps,i,init):
    lps[0] =init
    length = 0
    while i < M:
        if pat[i]!= pat[length]:
            if length != 0:
                xlen = length -1
                length = lps[xlen]
            else:
                i += 1
                lps[i-1] = 0
        else:
            i = i + 1
            j = i-1
            lps[j] = length + 1
            length += 1



def KMPSearch(pat, txt , i , j , M , N):
    #N = len(txt)
    #M = len(pat)

    lps = [0]*M

    computeLPSArray(pat, M, lps,1,0)


    while i < N:
        if j == M:
            a= i-j
            return np.arange(a, a+M, dtype=int).tolist()
            k=j-1
            j = lps[k]

        if pat[j] == txt[i]:
            i += 1
            j += 1

        elif i < N :
            if pat[j] != txt[i]:
                if j == 0:
                     i += 1
                else:
                    k=j-1
                    j = lps[k]

    return list()



df = pd.read_csv("./datasets/train/lcp_single_train.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df = addpd(df,"./datasets/trial/lcp_single_trial.tsv")
df = addpd(df,"./datasets/train/lcp_multi_train.tsv")
df = addpd(df,"./datasets/trial/lcp_multi_trial.tsv")
df = addpd(df,"./datasets/test/lcp_single_test.tsv")
df = addpd(df,"./datasets/test/lcp_multi_test.tsv")

df = df.reset_index(drop=True)


configuration = ElectraConfig()


torch.backends.cudnn.deterministic = True
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)
torch.cuda.manual_seed_all(123)

le = len(df)
rn = range(len(df))
Embed = {}
for i in tqdm(range(le)):
    df['sentence'][i] = "[CLS] " + df['sentence'][i] + " [SEP]"

    dftoken = df['token'][i]
    dfid = df['id'][i]
    word_token=tokenizer.tokenize(str(dftoken))
    dfsentence = df['sentence'][i]
    tokenized = tokenizer.tokenize(dfsentence)
    seglen = len(tokenized)
    segments_ids = [1] * seglen
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
    word_token_length = len(word_token)
    tokenizer_length = len(tokenizer.tokenize(df['sentence'][i]))
    token_indices = KMPSearch(word_token, tokenizer.tokenize(df['sentence'][i]),0,0,word_token_length, tokenizer_length)


    token_embeddings = torch.squeeze(getstates(torch,model,indexed_tokens, segments_ids), dim=0)

    embeddingnumarray = token_embeddings[0].numpy()
    embshape = embeddingnumarray.shape
    embedding = np.zeros(embshape, dtype = np.float32)
    if len(token_indices) > 0:
        for j in token_indices:
            tokemb = token_embeddings[j]
            lenx= len(token_indices)
            embedding += tokemb.numpy()
        temp = embedding/lenx
        embedding = temp
    Embed[dfid] = embedding.tolist()

with open("Electra_embeddings.json", "w") as outfile:
    json.dump(Embed, outfile, indent = 4)
