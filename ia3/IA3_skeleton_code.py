# Imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from typing import List
from seaborn import heatmap

# Plot styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 13

def getMaxes(arr: List, n: int) -> List:
    max_indices = []
    
    for i in range(n):
        max_val = -1
        max_idx = -1
        for j,v in enumerate(arr):
            if j not in max_indices and v > max_val:
                max_idx,max_val = j,v
                
        max_indices.append(max_idx)
        
    return max_indices


def topWords(train_data):
    # Split data into pos/neg data frames
    df_pos_tweets = train_data[train_data['sentiment'] == 1]
    df_neg_tweets = train_data[train_data['sentiment'] == 0]
    
    # Define vectorizer's and fit/transform to split data above
    pos_vectorizer = CountVectorizer(lowercase=True)
    neg_vectorizer = CountVectorizer(lowercase=True)
    pos_tweets_token_counts = pos_vectorizer.fit_transform(df_pos_tweets['text'])
    neg_tweets_token_counts = neg_vectorizer.fit_transform(df_neg_tweets['text'])
    
    # Get words in vectorizer
    pos_tweets_words = pos_vectorizer.get_feature_names_out()
    neg_tweets_words = neg_vectorizer.get_feature_names_out()

    # Get sum of counts
    sum_pos_tweets_token_counts = pos_tweets_token_counts.sum(axis=0).tolist()[0]
    sum_neg_tweets_token_counts = neg_tweets_token_counts.sum(axis=0).tolist()[0]
    
    # Find index values for 10 most frequent words
    pos_tweets_most_freq_indices = getMaxes(sum_pos_tweets_token_counts, 10)
    neg_tweets_most_freq_indices = getMaxes(sum_neg_tweets_token_counts, 10)
    
    print("\nPart 0.a : CountVectorizer")
    print("==============================")
    print("The 10 most frequent words in the positive comments: ")
    for i in pos_tweets_most_freq_indices:
        print("\"%s\" occurs %d times" % (pos_tweets_words[i], sum_pos_tweets_token_counts[i]))
                                          
    print("\nThe 10 most frequent words in the negative comments: ")
    for i in neg_tweets_most_freq_indices:
        print("\"%s\" occurs %d times" % (neg_tweets_words[i], sum_neg_tweets_token_counts[i]))
        
    # Fit TfidfVectorizer to split data
    pos_tfidfvectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
    neg_tfidfvectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
    pos_tweets_tfidf = pos_tfidfvectorizer.fit_transform(df_pos_tweets['text'])
    neg_tweets_tfidf = neg_tfidfvectorizer.fit_transform(df_neg_tweets['text'])
    
    # Get tweet words for each class
    pos_tweets_words = pos_tfidfvectorizer.get_feature_names_out()
    neg_tweets_words = neg_tfidfvectorizer.get_feature_names_out()

    # Find sum of word occurances
    sum_pos_tweets_tfidf = pos_tweets_tfidf.sum(axis=0).tolist()[0]
    sum_neg_tweets_tfidf = neg_tweets_tfidf.sum(axis=0).tolist()[0]

    # Find occurances of top 10 words
    pos_tweets_most_freq_indices = getMaxes(sum_pos_tweets_tfidf, 10)
    neg_tweets_most_freq_indices = getMaxes(sum_neg_tweets_tfidf, 10)

    print("\nPart 0.b : TF-IDF")
    print("==============================")
    print("The 10 most frequent words in the positive comments: ")
    for i in pos_tweets_most_freq_indices:
        print("\"%s\" occurs %d times" % (pos_tweets_words[i], sum_pos_tweets_tfidf[i]))

    print("\nThe 10 most frequent words in the negative comments: ")
    for i in neg_tweets_most_freq_indices:
        print("\"%s\" occurs %d times" % (neg_tweets_words[i], sum_neg_tweets_tfidf[i]))
        
    return


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
    n_train = len(y_train)
    n_val = len(y_val)
    
    svm = SVC(C=c, kernel=kernel, degree=deg, max_iter=25000, class_weight="balanced")
    svm.fit(X_train, y_train)
    
    y_pred_train = svm.predict(X_train)
    y_pred_val = svm.predict(X_val)
    
    acc_train = (n_train - np.count_nonzero(y_pred_train - y_train)) / n_train
    acc_val = (n_val - np.count_nonzero(y_pred_val - y_val)) / n_val
    
    return acc_train, acc_val, svm.n_support_


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
  
    
def plotNSV(exp, svms, title, pName):
    '''
    Plot number of support vectors for the SVM
    '''
    fig = plt.figure(figsize=(14,7))

    for i, (_, svs) in zip(exp, svms.items()):
        sv = sum(svs)
        plt.scatter(i, sv, s=sv*0.05)

    plt.xlabel("i")
    plt.ylabel("Support Vector Counts")
    plt.title("Number of SV\'s for c = 10^i | " + title)
    plt.savefig(pName)
    return

def optimizeC(X_train, y_train, X_val, y_val, l, r, kernel, deg=3):
    acc_val = {}
    acc_train = {}
    SVs = {}

    max_depth = 11
    cur_depth = 0
    
    while cur_depth <= max_depth:
        print("Optimizing C, iteration", cur_depth+1)
        if l not in acc_val.keys():
            acc_train[l], acc_val[l], SVs[l] = trainSVM(X_train, y_train, X_val, 
                                                         y_val, l, kernel, deg)

        if r not in acc_val.keys():
            acc_train[r], acc_val[r], SVs[r] = trainSVM(X_train, y_train, X_val, 
                                                         y_val, r, kernel, deg)

        m = (l + r) / 2
        acc_train[m], acc_val[m], SVs[m] = trainSVM(X_train, y_train, X_val, 
                                                     y_val, m, kernel, deg)

        acc_val_max = max(acc_val[m], acc_val[r], acc_val[l])

        if acc_val_max == acc_val[m]:
            l = (m + l) / 2
            r = (m + r) / 2
        elif acc_val_max == acc_val[l]:
            r = m
        else:
            l = m

        cur_depth += 1
    
    return acc_train, acc_val, SVs

def plotOptC(accDict, title, pName, xlim, ylim):
    '''
    Plot accuracy dictionary as a function of optimal c search
    '''
    if xlim != None:
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(1,2,1)
        ax1.scatter(list(accDict.keys()), list(accDict.values()))
        plt.xlabel("C")
        plt.ylabel(title + " accuracy")
        plt.title(title + " accuracy in regards to c (Optimized)")
    
        ax2 = plt.subplot(1,2,2)
        ax2.set(xlim=xlim, ylim=ylim)
        ax2.scatter(list(accDict.keys()), list(accDict.values()))
        plt.title("Zoomed in")
    else:
        plt.figure(figsize=(6,6))
        plt.scatter(list(accDict.keys()), list(accDict.values()))
        plt.xlabel("C")
        plt.ylabel(title + " accuracy")
        plt.title(title + " accuracy in regards to c (Optimized)")
        
    plt.savefig(pName)
    
    return

def trainSVMrbf(X_train, y_train, X_val, y_val, c, g):
    '''
    Train SVM with rbf using the passed params for c and gamma
    Return train accuracy, val accuracy, and number of support vectors
    '''
    n_val = len(y_val)
    n_train = len(y_train)
    
    svm = SVC(C=c, kernel="rbf", gamma=g ,max_iter=25000, class_weight="balanced")
    svm.fit(X_train, y_train)

    y_pred_train = svm.predict(X_train)
    y_pred_val = svm.predict(X_val)

    acc_train = (n_train - np.count_nonzero(y_pred_train - y_train)) / n_train
    acc_val = (n_val - np.count_nonzero(y_pred_val - y_val)) / n_val
    return acc_train, acc_val, svm.n_support_

def plotHeatMap(accMat, title, pName):
    '''
    Plot heatmap on accuracy as a function of gamma and c
    '''
    plt.figure(figsize=(11,4))
    ax = heatmap(accMat, annot=True, fmt=".4f", linewidth=0.2, cmap="crest")
    ax.set(xlabel="Gamma", ylabel="C")
    plt.yticks(rotation=0)
    plt.title(title + " Accuracy in regards to C and Gamma")
    plt.savefig(pName)
    return

def SVplot(xrange, SVc, title, pName):
    '''
    Plot number of support vectors for fixed parameter across another parameter.
    '''
    fig = plt.figure(figsize=(14,7))

    for i, svs in zip(xrange, SVc):
        plt.scatter(i, svs, s=svs*0.05)

    plt.xlabel(title)
    plt.ylabel("Support Vector Counts")
    plt.title("Number of SV\'s for 10^i | Function of " + title)
    plt.savefig(pName)
    return

        
def main():
    # Import training data
    df_train = pd.read_csv("IA3-train.csv")
    df_val = pd.read_csv("IA3-dev.csv")
    topWords(df_train)
    
    print("\n---------- PART 0: Preprocessing ----------\n")
    # ----------------- Generate vectorizer & form data splits ----------------- 
    tfidfvectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
    
    # Generate training data (sparse)
    X_train = tfidfvectorizer.fit_transform(df_train['text'])
    y_train = df_train['sentiment']

    # Generate validation data (sparse)
    X_val = tfidfvectorizer.transform(df_val['text'])
    y_val = df_val['sentiment']

    # Define initial c values
    cVals = [10**i for i in range(-4,5)]
    
    # Define dictionaries to hold training and acc values
    acc_train = {}
    acc_val = {}
    nSVs = {}
    
    print("\n---------- PART I: Linear SVM ----------\n")
    # ------------- Train model with linear kernel over initial c (10^-4,...,10^4) ------------- 
    print("\n ----- Training SVM with Linear Kernel ----- ")
    for c in cVals:
        print("Training on c =", c)
        acc_train[c], acc_val[c], nSVs[c] = trainSVM(X_train, y_train, X_val, y_val, c, "linear")
        
    # Plot accuracy for training and validation, plot number of SV's (across each c value)
    xrange = range(-4,5)
    plotAcc(xrange, acc_train, acc_val, "Linear SVM", "lsvmAcc.jpg")
    plotNSV(xrange, nSVs, "Linear SVM", "lsvmNSV.jpg")
    
    # ADD HERE (SEARCH FOR OPTIMAL C, PLOTS, ETC. FOR LINEAR KERNEL)
    tmpQTA, tmpQVA, tmpQSVs = optimizeC(X_train, y_train, X_val, y_val, 0.1, 1, "linear")
    plotOptC(tmpQTA, "Training", "optCtrainL.jpg", None, None)
    plotOptC(tmpQVA, "Validation", "optCvalL.jpg", None, None)
    
    print("\n---------- PART II: Quadratic SVM ----------\n")
    # ------------- Train model with quadratic kernel over initial c (10^-4,...,10^4) ------------- 
    print("\n ----- Training SVM with Quadratic Kernel ----- ")
    for c in cVals:
        print("Training on c =", c)
        acc_train[c], acc_val[c], nSVs[c] = trainSVM(X_train, y_train, X_val, y_val, c, "poly", 2)
    
    # Plot accuracy for training and validation, plot number of SV's (across each c value)
    plotAcc(xrange, acc_train, acc_val, "Quadratic SVM", "qsvmAcc.jpg")
    plotNSV(xrange, nSVs, "Quadratic SVM", "qsvmNSV.jpg")
    
    # Search for optimal value of c, plot training and validation accuracy for all values
    tmpQTA, tmpQVA, tmpQSVs = optimizeC(X_train, y_train, X_val, y_val, 0.3, 0.7, "poly", 2)
    plotOptC(tmpQTA, "Training", "optCtrain.jpg", None, None)
    plotOptC(tmpQVA, "Validation", "optCval.jpg", (0.5, 0.55), (0.916, 0.918))    
    
    print("\n---------- PART III: SVM with RBF kernel ----------\n")
    # --------- Train model with rbf kernel over initial c (10^-4,...,10^4) and gamma (10^-5,...,10^1) --------- 
    print("\n ----- Training SVM with RBF Kernel ----- ")
    gRange = [10**i for i in range(-5,2)]
    cRange = [10**i for i in range(-4,5)]

    for c in cRange:
        for g in gRange:
            print("Training on c =", c, "| gamma =", g)
            acc_train[(c,g)], acc_val[(c,g)], nSVs[(c,g)] = trainSVMrbf(X_train, y_train, X_val, y_val, c, g)
            
    # Plot heatmap as a function of c and gamma for training and validation accuracy
    # Convert dictionaries for training and validation accuract to 2D matrix (data frame) with x=gamma, y=c
    tAccMat = pd.DataFrame(np.nan, index=cRange, columns=gRange)
    for i,c in enumerate(cRange):
        newRow = [acc_train[(c, g)] for g in gRange]
        tAccMat.iloc[i] = newRow

    vAccMat = pd.DataFrame(np.nan, index=cRange, columns=gRange)
    for i,c in enumerate(cRange):
        newRow = [acc_val[(c, g)] for g in gRange]
        vAccMat.iloc[i] = newRow
        
    plotHeatMap(tAccMat, "Training", "rbfHeatTrain.jpg")
    plotHeatMap(vAccMat, "Validation", "rbfHeatVal.jpg")
    
    # Plot support vectors as a function of c with fixed gamma = 0.1
    fixedG = [sum(nSVs[(c, 0.1)]) for c in cRange]
    SVplot(range(-4,5), fixedG, "C (Gamma = 0.1)", "cSVcount.jpg")
    
    # Plot support vectors as a function of gamma with fixed c = 10
    fixedC = [sum(nSVs[(10, g)]) for g in gRange]
    SVplot(range(-5,2), fixedC, "Gamma (C=10)", "gammaSVcount.jpg")
    
    # Optimize c and gamma value (optimal model)
    opt_tAcc = {}
    opt_vAcc = {}

    gRange = np.arange(0.005, 0.041, 0.005)
    cRange = np.arange(9, 11.1, 0.25)

    for c in cRange:
        for g in gRange:
            g = round(g, 3) # fix floating point error
            print("Training on c =", c, "| gamma =", g)
            opt_tAcc[(c,g)], opt_vAcc[(c,g)], _ = trainSVMrbf(X_train, y_train, X_val, y_val, c, g)
    
    # Convert accuracy for optimal model into 2D array, then plot heatmap
    gColNames = [round(g,3) for g in gRange]
    tAccMat = pd.DataFrame(np.nan, index=cRange, columns=gColNames)
    for i,c in enumerate(cRange):
        newRow = [opt_tAcc[(c, round(g,3))] for g in gRange]
        tAccMat.iloc[i] = newRow

    vAccMat = pd.DataFrame(np.nan, index=cRange, columns=gColNames)
    for i,c in enumerate(cRange):
        newRow = [opt_vAcc[(c, round(g,3))] for g in gRange]
        vAccMat.iloc[i] = newRow

    plotHeatMap(tAccMat, "Training", "rbfHeatOptTrain.jpg")
    plotHeatMap(vAccMat, "Validation", "rbfHeatOptVal.jpg")

if __name__ == "__main__":
    main()
