import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, cluster
from sklearn.svm import SVC
import re
from string import punctuation

from GloVe_Embedder import GloVe_Embedder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Plot styling
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.style.use('ggplot')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 11

# Adapted from IA3
def trainSVM(X_train, y_train, X_val, y_val, c, kernel, deg=3):
    """
    Description: scikit learn linearSVC wrapper
    Param:
        X_train  [in]: training data
        y_train  [in]: training label
        X_val    [in]: validation data
        y_val    [in]: validation label
        c        [in]: Regularization parameter. The strength of the 
                       regularization is inversely proportional to C. Must be 
                       strictly positive.
        kernel   [in]: Kernel type for SVC
        deg      [in]: degree (only for poly kernel) - if not poly, this is 
                       ignored (default param set to sklearn default = 3)
    Return: training accuracy, validation accuracy, and number of SV's (respectively)
    """
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    svm = SVC(C=c, kernel=kernel, degree=deg, max_iter=25000)
    svm.fit(X_train, y_train)
    
    y_pred_train = svm.predict(X_train)
    y_pred_val = svm.predict(X_val)
    
    acc_train = (n_train - np.count_nonzero(y_pred_train - y_train)) / n_train
    acc_val = (n_val - np.count_nonzero(y_pred_val - y_val)) / n_val
    
    return acc_train, acc_val, svm.n_support_

# Build data set of the 5 seed words + the 29 closest words based on distance.
# Returns two data frames: d150embd (word embeddings for each word)
#                          d150wrds (base words corresponding to each embedding, cluster num (seed word position))
def buildData(GV_embedded):
    seed_words = ["flight", "good", "terrible", "help", "late"]

    d150embd = [] # build data set of word vectors
    d150wrds = [] # build data set of words (raw)

    seed_words_embeddings = GV_embedded.embed_list(seed_words)
    clusterNum = 1

    for word in seed_words:
        embedding = GV_embedded.embed_str(word)
        d150embd.append(embedding) # append embedding of words
        d150wrds.append([word, int(clusterNum)]) # append word & cluster number (used for graphing)
        cluster = GV_embedded.find_k_nearest(embedding, 29)
        print(word, "cluster:")

        for w in cluster:
            print("\t", w[0], w[1])
            d150embd.append(GV_embedded.embed_str(w[0])) # append embedding of current words
            d150wrds.append([w[0], int(clusterNum)]) # append word & cluster number (used for graphing)
        clusterNum += 1
        print()

    return d150embd, d150wrds

# Plot scatter of 2d PCA representation of word embeddings (colored by seed word)
def plot_2d_cluster(X, d150, label, fName):
    plt.figure(figsize=(10,5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=np.asarray(d150[:,1], dtype=int), cmap="Set1")
    plt.legend(handles=scatter.legend_elements()[0], labels=["flight", "good", "terrible", "help", "late"])
    plt.title(label)
    plt.ylabel("d2")
    plt.xlabel("d1")
    plt.savefig(fName)
    return

# Fit embeddings to tSNE (two dimensions across perplexity range). Plot scatter plot 
# for each fitting of the data to analyze clusters.
def plottSNE(pRange, embd, wrds, fName):
    fig, axs = plt.subplots(5, 3, figsize=(14, 16), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()
    for i, p in enumerate(pRange):
        X_embd = TSNE(n_components=2, perplexity=p).fit_transform(np.asarray(embd))
        title = "t-SNE (Parity = " + str(p) + ")" 
        axs[i].scatter(X_embd[:, 0], X_embd[:, 1], c=np.asarray(wrds[:,1], dtype=int), cmap="Set1")
        axs[i].set_title(title)
    fig.savefig(fName)
    return

# Calculate purity for given kmeans model
def purity(y_true, y_pred):
    cMat = cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cMat, axis=0)) / np.sum(cMat)

# Plotting clustering evaluation metric (rand score, inertia, mutual info, purity)
def clusteringPlots(k, rS, iS, miS, pS, fName):
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(k, rS)
    plt.title("Rand Score")
    plt.subplot(2,2,2)
    plt.plot(k, iS)
    plt.title("Inertia")
    plt.subplot(2,2,3)
    plt.plot(k, miS)
    plt.title("Mutual Info Score")
    plt.subplot(2,2,4)
    plt.plot(k, pS)
    plt.title("Purity Score")
    plt.savefig(fName)
    return

def average_embeddings(ge, sentence):
    emb = ge.embed_list(sentence.split())
    
    return emb.sum(axis=0) / len(emb)

# Adapted from IA3
def plotAcc(exp, tAcc, vAcc, title, pName):
    '''
    Plot the accuracy for train and validation data set for SVM
    '''
    fig = plt.figure(figsize=(14,7))
    plt.plot(exp, list(tAcc.values()), label="Training")
    plt.plot(exp, list(vAcc.values()), label="Validation")
    
    plt.xlabel("i")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for c = 10^i | " + title)
    plt.legend()
    plt.savefig(pName)
    return

# Adapted from https://www.kaggle.com/code/artemzapara/twitter-feeds-classification-with-glove-embeddings/notebook
def clean_text(text):
    text = text.lower() # lowercase
    text = re.sub('@\w+ ', ' ', text)    # Remove "@..." which are not related to the sentiment
    text = re.sub('[%s]' % re.escape(punctuation), '', text) # Remove punctuation
    return text

def main():
    # ------------------------------------ EXPLORE WORD EMBEDDINGS ------------------------------------
    
    # Build data set of 150 words closeset to the 5 seed words (29 each + seed words)
    ge = GloVe_Embedder("GloVe_Embedder_data.txt")
    
    # print("Seed words & their 29 closest neighbors (by distance)...")
    # d150embd, d150wrds = buildData(ge)
    
    # # Perform PCA to reduce embedding to 2 dimensions, plot as scatter (colored by cluster)
    # print("\nPerforming PCA - reducing and plotting embeddings into 2D...")
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(d150embd)
    # d150wrds = np.asarray(d150wrds)
    # plot_2d_cluster(X_2d, d150wrds, "PCA on 150 Clustered Words", "pcaClusters.png")
    
    # # Fit and plot tSNE across perplexity values
    # print("\nFitting and plotting tSNE across perplexity range [3,63]...")
    # plottSNE([x for x in range(3,18)], d150embd, d150wrds, "sne_exploration1.png")
    # plottSNE([x for x in range(18,33)], d150embd, d150wrds, "sne_exploration2.png")
    # plottSNE([x for x in range(33,48)], d150embd, d150wrds, "sne_exploration3.png")
    # plottSNE([x for x in range(48,63)], d150embd, d150wrds, "sne_exploration4.png")
    
    # # Clustering word embeddings with Kmeans
    # print("\nFitting and plotting kmeans across cluster range [2,20]...")
    # purityScore = []
    # randScore = []
    # inertiaScore = []
    # mutualInfoScore = []

    # kRange = [x for x in range(2,21)]

    # for k in kRange:
    #     kmeans = KMeans(n_clusters=k).fit(d150embd)
    #     predsLabels = kmeans.labels_
    #     purityScore.append(purity(d150wrds[:,1], predsLabels))
    #     randScore.append(adjusted_rand_score(d150wrds[:,1], predsLabels))
    #     inertiaScore.append(kmeans.inertia_)
    #     mutualInfoScore.append(normalized_mutual_info_score(d150wrds[:,1], predsLabels))
    
    # clusteringPlots(kRange, randScore, inertiaScore, mutualInfoScore, purityScore, "kmeansPlots.png")

    # --------------------------------- WORD EMBEDDINGS FOR CLASSIFICATION ---------------------------------
    
    ## ------------------- Approach 1: Weighted Average Embeddings ---------------
    df_train = pd.read_csv("IA3-train.csv")
    df_val = pd.read_csv("IA3-dev.csv")
    
    X_train_aemb = []
    for tweet in df_train['text']:
        avg_emb = average_embeddings(ge, tweet)
        X_train_aemb.append(avg_emb)
        
    X_val_aemb = []
    for tweet in df_val['text']:
        avg_emb = average_embeddings(ge, tweet)
        X_val_aemb.append(avg_emb)
    
    df_X_train_aemb = pd.DataFrame(X_train_aemb)
    y_train = df_train['sentiment']

    df_X_val_aemb = pd.DataFrame(X_val_aemb)
    y_val = df_val['sentiment']

    # Define initial c values
    cVals = [10**i for i in range(-4,5)]

    acc_train = {}
    acc_val = {}
    nSVs = {}
        
    for c in cVals:
        print("Training on c =", c)
        acc_train[c], acc_val[c], nSVs[c] = trainSVM(df_X_train_aemb, y_train, df_X_val_aemb, y_val, c, "linear")
        
    xrange = range(-4,5)
    plotAcc(xrange, acc_train, acc_val, "Linear SVM", "lsvmAcc.png")
    
    ## ------------------- Approach 2: Clean the tweets ---------------
    df_train_clean = df_train
    df_val_clean = df_val

    # Applying the cleaning function to both test and train datasets
    df_train_clean['text'] = df_train['text'].apply(lambda x: clean_text(x))
    df_val_clean['text'] = df_val['text'].apply(lambda x: clean_text(x))
    
    X_train_aemb = []
    tweet_vocab = set()
    for tweet in df_train_clean['text']:
        new_vocab = set(tweet.split())
        avg_emb = average_embeddings(ge, tweet)
        X_train_aemb.append(avg_emb)
        tweet_vocab = tweet_vocab.union(new_vocab)
        
    X_val_aemb = []
    for tweet in df_val_clean['text']:
        avg_emb = average_embeddings(ge, tweet)
        X_val_aemb.append(avg_emb)
    
    df_X_train_aemb = pd.DataFrame(X_train_aemb)
    df_X_val_aemb = pd.DataFrame(X_val_aemb)
    acc_train = {}
    acc_val = {}
    nSVs = {}
        
    for c in cVals:
        print("Training on c =", c)
        acc_train[c], acc_val[c], nSVs[c] = trainSVM(df_X_train_aemb, y_train, df_X_val_aemb, y_val, c, "linear")
        
    plotAcc(xrange, acc_train, acc_val, "Linear SVM", "lsvmAcc.png")

if __name__ == "__main__":
    main()
