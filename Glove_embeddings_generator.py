import numpy as np
import pandas as pd
from tqdm import tqdm
import csv, json

# Here we store glove embeddings of tokens in single and multi datasets
# Loading Embeddings
embeddings_dict1 = {}
with open("glove.6B.200d.txt", 'r',  encoding="utf8") as f:
    for line in f:
        values = line.split()
        embeddings_dict1[values[0]] = np.asarray(values[1:], "float32")

def getEmbeddingSingle(Embed,df):
    tokens = df['token']

    for i in tqdm(range(len(df))):
        token = str(tokens[i]).lower()
        if token not in embeddings_dict1.keys():
            embedding = np.zeros(200)
        else:
            embedding = embeddings_dict1[token]
        list_embed = embedding.tolist()
        Embed[df['id'][i]] = list_embed
    return Embed

def getEmbeddingMulti(Embed,df):
    tokens = df['token']

    for i in tqdm(range(len(df))):
        token1 = str(tokens[i]).split()[0].lower()
        token2 = str(tokens[i]).split()[1].lower()
        if token1 in embeddings_dict1.keys() and token2 in embeddings_dict1.keys():
            embedding = (0.5 * embeddings_dict1[token1]) + (0.5 * embeddings_dict1[token2])
        else:
            embedding = np.zeros(200)
        list_embed = embedding.tolist()
        Embed[df['id'][i]] = list_embed
    return Embed

Embed = {}

#Add single embeddings
df_train = pd.read_csv("./datasets/train/lcp_single_train.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_trial = pd.read_csv("./datasets/trial/lcp_single_trial.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_test = pd.read_csv("./datasets/test/lcp_single_test.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')

Embed = getEmbeddingSingle(Embed,df_train)
Embed = getEmbeddingSingle(Embed,df_trial)
Embed = getEmbeddingSingle(Embed,df_test)

#Add multi embeddings
df_train = pd.read_csv("./datasets/train/lcp_multi_train.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_trial = pd.read_csv("./datasets/trial/lcp_multi_trial.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_test = pd.read_csv("./datasets/test/lcp_multi_test.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')

Embed = getEmbeddingMulti(Embed,df_train)
Embed = getEmbeddingMulti(Embed,df_trial)
Embed = getEmbeddingMulti(Embed,df_test)

with open("glove_embeddings.json", "w") as outfile:  
    json.dump(Embed, outfile, indent = 4)
