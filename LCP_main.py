import numpy as np
import pandas as pd
import csv
import json
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 

original_to_standard = {0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4}
standard_to_original = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1}
dfstring = []


#Loading Embeddings
f = open('glove_embeddings.json') 
embeddings_dict1 = json.load(f) 

f = open('Electra_embeddings.json') 
embeddings_dict2 = json.load(f) 

def electra_train(df, embeddings_dict):
    tokens = df['id'].values
    complexity = df['complexity'].values

    x_train = []
    for i in range(len(tokens)):
        tokenx = tokens[i]
        embdict = embeddings_dict[tokenx]
        embdictarray = np.array(embdict)
        x_train.append(embdictarray)

    y_train = complexity
    model = LinearRegression().fit(np.array(x_train, dtype=np.float32), y_train)        
    return model

strtoke = []
def electraPredictions(df, model, embeddings_dict):
    dfcomp = df['complexity']
    complexity = dfcomp.values
    x_test = []
    dfid = df['id']
    tokens = dfid.values
    
    lentok = len(tokens)
    rn = range(lentok)
    for i in range(lentok):
        tokenx = tokens[i]
        embdict = embeddings_dict[tokenx]
        temp = np.array(embdict)
        x_test.append(temp)
        nparrayval = np.array(x_test, dtype=np.float32)

    x_test = nparrayval
    y_pred = model.predict(x_test)
    return y_pred
    
#Getting training feature vector x_train from embedding
def getx_train(tokens, embeddings_dict):
    x_train = []
    toklen = len(tokens)
    for i in range(toklen):
        token = tokens[i]
        emb = embeddings_dict[token]
        temp = np.array(emb)
        x_train.append(temp)

    x_train = np.array(x_train, dtype = np.float32)
    return x_train

def generateOutput(df_train,df_test,embeddings_dict1,embeddings_dict2,w1,w2,outfile):
    df_test['complexity'] = 0.00000
    df_test['Predicted Complexity'] = 0.00000
    df_str_token = ""
    predictions = combinedPredictions(df_train,df_test, embeddings_dict1, embeddings_dict2)
    lentest = len(df_test)
    rntest = range(lentest)
    y_pred = (w1 * predictions[0]) + (w2 * predictions[1])
    
    for i in rntest:    
        df_test['Predicted Complexity'][i] = y_pred[i]

    print(y_pred)
    print("Generating output file: ")
    df_final = df_test.drop(columns=['complexity','corpus','sentence','token'])
    df_final.to_csv(outfile,index=False,header=None)
    print("Output file {filename} generated".format(filename = outfile))

#Predictions for Glove
def glovePredictions(df_train,df_test, embeddings_dict1, n=20):
    x_train = getx_train(df_train['id'], embeddings_dict1)
    x_test = getx_train(df_test['id'], embeddings_dict1)
    y_test = df_test['complexity']
    lenytest = len(y_test)
    yshapezero = np.zeros(y_test.shape,dtype=float)

    #Generate class annotations for scores in dataset
    scores = df_train['complexity']
    annotations = []
    for i in range(len(scores)):
        low = 0
        while scores[i] >= (low + 0.25):
            low += 0.25
        high = low + 0.25
        temp = int(original_to_standard[high])*np.ones(n,dtype=int)
        alpha = (high - scores[i])/(high-low)
        num_low = int(np.floor(alpha*n))
        for i in range(num_low):
            ogtostd = original_to_standard[low]
            ogtostdint = int(ogtostd)
            temp[i]=ogtostdint
        templist = temp.tolist()
        annotations.append(templist)
    annotations = np.array(annotations)

    #Get predictions for n number of passes
    predictions = []
    for i in range(n):
        y_train = annotations[:,i]
        model = SVC(kernel = 'rbf', verbose=1,C = 1).fit(x_train, y_train)
        predictions.append(model.predict(x_test))
    
    #Get scores from classes
    pred = np.array(predictions, dtype = float)
    predlen = len(pred)
    predrange = range(predlen)
    for i in range(0,predlen):
        predlenx = len(pred[i])
        for j in range(predlenx):
            predij = pred[i,j]
            predint = int(pred[i,j])
            stdtoog = standard_to_original[predint]
            pred[i,j]= stdtoog
    
    
    #Get mean of scores across passes
    y_pred = yshapezero
    for j in range(lenytest):
        predmean = pred[:,j].mean()
        y_pred[j] = predmean
    return y_pred

# Gives predictions of both models
def combinedPredictions(df_train,df_test, embeddings_dict1, embeddings_dict2):
    # Training Electra Model
    model = electra_train(df_train, embeddings_dict2)
    dfpredtemp=[]
    predictions = []

    # Electra
    predictions.append(electraPredictions(df_test, model, embeddings_dict2))

    # Glove
    predictions.append(glovePredictions(df_train,df_test, embeddings_dict1, 50))
    return predictions

##################################################################################################################
# Single Words
df_train = pd.read_csv("./datasets/train/lcp_single_train.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_trial = pd.read_csv("./datasets/trial/lcp_single_trial.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_test = pd.read_csv("./datasets/test/lcp_single_test.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')

df_train = df_train.append(df_trial)
df_train = df_train.reset_index(drop=True)

print(df_train.shape)
generateOutput(df_train,df_test,embeddings_dict1,embeddings_dict2,0.5,0.5,'single_test_predictions.csv')

##################################################################################################################
# Multiple Words
df_train = pd.read_csv("./datasets/train/lcp_multi_train.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_trial = pd.read_csv("./datasets/trial/lcp_multi_trial.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_test = pd.read_csv("./datasets/test/lcp_multi_test.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')

#using part of single word training data too for training
df_curr = pd.read_csv("./datasets/train/lcp_single_train.tsv",delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
df_sample = df_curr.sample(frac=1,random_state=2)
df_sample = df_sample[:1900]

df_train = df_train.append(df_trial)
df_train = df_train.append(df_sample)
df_train = df_train.reset_index(drop=True)

generateOutput(df_train,df_test,embeddings_dict1,embeddings_dict2,0.5,0.5,'multi_test_predictions.csv')
