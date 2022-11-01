# AI534 IA2, Group 50: Vy Bui, Sebastian Mueller, Derek Helms

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

# Plot styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 13
plt.ioff()

# Define dictionaries (training data vals for normalizing)
mu_dict = {}
sigma_dict = {}

# Loads a data file from a provided file location.
def load_data(path):
    return pd.read_csv(path)
    
# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.
def preprocess_data(data):
    numFeats = ["Age", "Annual_Premium", "Vintage"]
    preprocessed_data = data.copy() # copy so raw data not altered
    
    if len(mu_dict) == 0:
        for col in numFeats:
            mu_dict[col] = data[col].mean()
            sigma_dict[col] = data[col].std()
            
    for col in numFeats:
        preprocessed_data[col] = ((preprocessed_data[col] - mu_dict[col])/sigma_dict[col])
        
    return preprocessed_data

# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def sigmoid(logodds: np.ndarray):
    return 1 / (1 + np.exp(-logodds))

def LR_L2_train_torch(train_data, val_data, _lambda, alpha, isNoisy):
    device = torch.device('mps')
    vsigmoid = torch.nn.Sigmoid()
    
    features = train_data.columns.drop("Response")
    X_train = train_data[features]
    y_train = train_data["Response"].to_frame()
    
    X_val = val_data[features]
    y_val = val_data["Response"].to_frame()
    
    y_train = y_train.rename(columns={"Response": 0})
    y_val = y_val.rename(columns={"Response": 0})
    
    n = len(X_train)
    n_features = len(X_train.columns.values)
    
    # Found through experimenting (might need to be changed for large lambda?)
    if isNoisy:
        epsilon = 0.0000000002
    else:
        epsilon = 0.0000000001
            
    # TODO: randomized?
    w = (np.ones(n_features) * 0.2).reshape(n_features,1)
    acc_train = []
    acc_val = []
    grad_vals = []
    
    converged = False
    iter = 1
    
    w = torch.tensor(w,dtype=torch.float32,device=device)
    X_train = torch.tensor(X_train.values,dtype=torch.float32,device=device)
    y_train = torch.tensor(y_train.values,dtype=torch.float32,device=device)
    X_val = torch.tensor(X_val.values,dtype=torch.float32,device=device)
    y_val = torch.tensor(y_val.values,dtype=torch.float32,device=device)
    
    while not converged:
        log_odds = vsigmoid(torch.tensordot(X_train, w, dims=1))
        grad = torch.mul(torch.tensordot((y_train - log_odds).T, X_train, dims=1), alpha / n)
        w += grad.T
        
        # Extra info
        # log_loss_pos = y_train.T.dot(np.log(log_odds))[0][0]
        # log_loss_neg = (1 - y_train.T).dot(1 - np.log(log_odds))[0][0]
        # log_loss_avg = -(log_loss_pos + log_loss_neg) / n
        # reg = _lambda * np.linalg.norm(w) ** 2
        # log_loss = log_loss_avg + reg
        # log_losses.append(log_loss_avg)
        # grad_l2 = np.linalg.norm(grad.values)
        # w_l2 = np.linalg.norm(w.values)
        
        # Regularization
        w0 = w[0][0]     # Exclude w0
        w -= (alpha * _lambda) * w
        w[0][0] = w0
        
        y_pred_train = y_train - (vsigmoid(torch.tensordot(X_train, w, dims=1)) >= 0.5).int()
        acc = float((y_pred_train == 0).sum()) / n
        acc_train.append(acc)
        
        y_pred_val = y_val - (vsigmoid(torch.tensordot(X_val, w, dims=1)) >= 0.5).int()
        acc = float((y_pred_val == 0).sum()) / len(y_val)
        acc_val.append(acc)
        
        # Check for convergence
        magLw = float(torch.linalg.norm(grad) / n) # magnitude of gradient (change in weights)
        grad_vals.append(magLw)
        
        if iter % 1000 == 0:
            print("Iteration 1000, magLw = %f, change rate = %f" % (magLw, abs(grad_vals[-1] - grad_vals[-2])))
        
        # Check for difference between cur and past gradient, if minimal change then model has converged
        if (len(grad_vals) > 2) and (abs(grad_vals[-1] - grad_vals[-2]) < epsilon):
            print("learning_rate = %f, lambda = %f, converged at iter #%d" % (alpha, _lambda, iter))
            converged = True # set model to converged, return weights and train/val acc
            
        iter += 1
    
    return w.cpu().numpy(), acc_train, acc_val

def LR_L2_train(train_data, val_data, _lambda, alpha, isNoisy):
    features = train_data.columns.drop("Response")
    X_train = train_data[features]
    y_train = train_data["Response"].to_frame()
    
    X_val = val_data[features]
    y_val = val_data["Response"].to_frame()
    
    y_train = y_train.rename(columns={"Response": 0})
    y_val = y_val.rename(columns={"Response": 0})
    
    n = len(X_train)
    n_features = len(X_train.columns.values)
    
    # Found through experimenting (might need to be changed for large lambda?)
    if isNoisy:
        epsilon = 0.0000000002
    else:
        epsilon = 0.0000000001
            
    # TODO: randomized?
    w = (np.ones(n_features) * 0.2).reshape(n_features,1)
    acc_train = []
    acc_val = []
    grad_vals = []
    
    converged = False
    iter = 1
    while not converged:
        log_odds = sigmoid(X_train.dot(w))
        grad = ((y_train - log_odds).T.dot(X_train)).T / n
        w += alpha * grad
        
        # Extra info
        # log_loss_pos = y_train.T.dot(np.log(log_odds))[0][0]
        # log_loss_neg = (1 - y_train.T).dot(1 - np.log(log_odds))[0][0]
        # log_loss_avg = -(log_loss_pos + log_loss_neg) / n
        # reg = _lambda * np.linalg.norm(w) ** 2
        # log_loss = log_loss_avg + reg
        # log_losses.append(log_loss_avg)
        # grad_l2 = np.linalg.norm(grad.values)
        # w_l2 = np.linalg.norm(w.values)
        
        # Regularization
        w0 = w[0][0]     # Exclude w0
        w -= (alpha * _lambda) * w
        w[0][0] = w0
        
        y_pred_train = y_train - (sigmoid(X_train.dot(w)).rename(columns={"Response":0}) >= 0.5).astype(int)
        acc = (y_pred_train == 0).sum()[0] / n
        acc_train.append(acc)
        
        y_pred_val = y_val - (sigmoid(X_val.dot(w)).rename(columns={"Response":0}) >= 0.5).astype(int)
        acc = (y_pred_val == 0).sum()[0] / len(X_val)
        acc_val.append(acc)
        
        # Check for convergence
        
        magLw = np.linalg.norm(grad.values) / n # magnitude of gradient (change in weights)
        grad_vals.append(magLw)
        # Check for difference between cur and past gradient, if minimal change then model has converged
        if (len(grad_vals) > 2) and (abs(grad_vals[-1] - grad_vals[-2]) < epsilon):
            print("learning_rate = %f, lambda = %f, converged at iter #%d" % (alpha, _lambda, iter))
            converged = True # set model to converged, return weights and train/val acc
            
        iter += 1
    
    return w, acc_train, acc_val

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, _lambda, lr): 
    # Seperate training and validation data into X and y data frames
    features = train_data.columns.drop("Response")
    X_train = train_data[features]
    y_train = train_data["Response"].to_frame()
    X_val = val_data[features]
    y_val = val_data["Response"].to_frame()
    
    # Initialize weights
    N = len(X_train) # number of observations
    d = len(X_train.columns.values) # number of features
    w_init = (np.ones(d)*0.2).reshape(d,1)
    weights = w_init
    
    # Lists containing training and validation accuracy at each iteration
    train_acc = []
    val_acc = []
    grad_vals = []
    threshold = 0.50 # threshold for predicting class (1 if >=, 0 if <)
    
    # Set flag to check for convergence (used to continue training)
    converged = False
    epsilon = 0.00005 # found through experimenting (might need to be changed for large lambda?)
    
    # Create sigmoid function
    sigmoid = lambda wTx: 1/(1+np.exp(-wTx)) 
    
    # Traing model while not converged
    while not converged:
        # Update weights
        gradW = X_train.T.dot(y_train - sigmoid(X_train.dot(weights)).rename(columns={0:"Response"}))
        weights = (weights + lr*(1/N)*gradW).values
        
        # Apply L1 regularization
        for j in range(1,d):
            weights[j] = (np.sign(weights[j]))*max(0,np.abs(weights[j])-lr*_lambda)
            
        # Train accuracy (before or after weights update?)
        ypredT = y_train - (sigmoid(X_train.dot(weights)).rename(columns={0:"Response"}) >= threshold).astype(int)
        tAcc = (ypredT == 0).sum()[0] / N
        train_acc.append(tAcc)
        
        # Val accuracy (before or after weights update?)
        ypredV = y_val - (sigmoid(X_val.dot(weights)).rename(columns={0:"Response"}) >= threshold).astype(int)
        vAcc = (ypredV == 0).sum()[0] / len(X_val)
        val_acc.append(vAcc)
            
        # Check for convergence 
        magLw = np.linalg.norm(gradW.values)/N # magnitude of gradient (change in weights)
        grad_vals.append(magLw)
        # Check for difference between cur and past gradient, if minimal change then model has converged
        if (len(grad_vals) > 2) and (abs(grad_vals[-1] - grad_vals[-2]) < epsilon):
            print("Model Converged, lambda:", _lambda)
            converged = True # set model to converged, return weights and train/val acc
    
    print("Learn Rate:", lr)
    print("Train Acc:", train_acc[-1])
    print("Val Acc:", val_acc[-1])
    print("")
    return weights, train_acc, val_acc

# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_losses(tAccDict, vAccDict, fName):
    fig = plt.figure(figsize=(16,6))
    
    # Plot training accuracy (left subplot)
    ax1 = plt.subplot(1,2,1)
    for lmbda, acc in tAccDict.items():
        plt.plot(np.arange(1,len(acc)+1), acc, label=lmbda, linewidth=2)
    plt.legend(title="Lambda")   
    plt.xlabel("Iterations")
    plt.title('Training Accuracy per Regularization Parameter')
    plt.ylabel('Accuracy')
    
    # Plot validation accuracy (right subplot)
    ax2 = plt.subplot(1,2,2)
    for lmbda, acc in vAccDict.items():
        plt.plot(np.arange(1,len(acc)+1), acc, label=lmbda, linewidth=2)
    plt.legend(title="Lambda")   
    plt.xlabel("Iterations")
    plt.title('Validation Accuracy per Regularization Parameter')
    # Show and save plot
    plt.tight_layout()
    fig.savefig(fName)
    
    return

def sparsity_graph(train_weights, pName):   
    plt.figure(figsize=(6,4))
    
    if type(train_weights[list(train_weights.keys())[0]]) == np.ndarray:
        nZeros = [(train_weights[lbl].reshape(197) <= 10**(-6)).sum() for lbl in train_weights.keys()]
    else:
        nZeros = [(train_weights[lbl].to_numpy().reshape(197) <= 10**(-6)).sum() for lbl in train_weights.keys()]
        
    lmbLabel = list(train_weights.keys())
    plt.bar(lmbLabel, nZeros, align='center', alpha=0.8)
    plt.ylabel('Number of Zero Features')
    plt.xlabel("Regularization (Lambda) Value")
    plt.savefig(pName)
    return

def plot_acc_per_lambda(trainAccLmbd, valAccLmbd, fName):
    acc_train = []
    acc_val = []
    _lambdas = []
    for i in trainAccLmbd.keys():
        acc_train.append(trainAccLmbd[str(i)][-1])
        acc_val.append(valAccLmbd[str(i)][-1])
        _lambdas.append(float(i))

    fig2 = plt.figure(figsize=(8,6))
    
    plt.scatter(np.log10(_lambdas), acc_train, label="Train accuracy")
    plt.scatter(np.log10(_lambdas), acc_val, label="Validation accuracy")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams["legend.edgecolor"] = 'black'
    plt.rcParams["legend.fontsize"] = 10
    plt.legend()
    plt.savefig(fName)
    
    return


# **REMOVE LATER** (handled in preprocess_data skeleton code function)
def normalize(data: pd.DataFrame, column: str, mu: float, sigma: float):
    return (data[column] - mu)/sigma

def sigmoid(logodds: np.ndarray):
    return 1 / (1 + np.exp(-logodds))

def main():
    print("\n---------- PART I: Logistic regression with L2 (Ridge) regularization ----------\n")
    df_train = pd.read_csv("IA2-train.csv")
    df_val = pd.read_csv("IA2-dev.csv")

    print("\n---------- PREPROCESSING ----------\n")
    numerical_feas = ["Age", "Annual_Premium", "Vintage"]

    df_norm_train = pd.DataFrame(df_train)
    df_norm_val = pd.DataFrame(df_val)

    for col in numerical_feas:
        mu = df_train[col].mean()
        sigma = df_train[col].std()

        df_norm_train[col] = normalize(df_train, col, mu, sigma)
        df_norm_val[col] = normalize(df_val, col, mu, sigma)

    print("\n---------- TRAINING ----------\n")
    _lambdas = [10**(x) for x in range(-4,3)] # generate values of lambda 10^i, i in [-4,2]

    # Learning rates that work best for corresponding lambda
    lrs = [0.05, 0.04, 0.03, 0.035, 0.08, 0.001, 0.0001, 0.000004]

    acc_train_lmb = {} # dictionary to store train acc at each iteration for all models, indexed by lambda
    acc_val_lmb = {} # dictionary to store val acc at each iteration for all models, indexed by lambda
    w_train_lmb = {} # dictionary to store weights for all models, indexed by learning rate

    for lmbd, lr in zip(_lambdas, lrs):
        w,acc_train,acc_val = LR_L2_train(df_norm_train, df_norm_val, lmbd, lr, False) # train model with current lambda value
        acc_train_lmb[str(lmbd)] = acc_train # store training accuracy, indexed by lambda value
        acc_val_lmb[str(lmbd)] = acc_val # store val accuracy, indexed by lambda value
        w_train_lmb[str(lmbd)] = w # store final training weights, indexed by lambda value

    plot_losses(acc_train_lmb, acc_val_lmb, "L2_accuracy_per_iteration.jpg")
    plot_acc_per_lambda(acc_train_lmb, acc_val_lmb, "L2_accuracy_per_lambda.jpg")
    sparsity_graph(w_train_lmb, "L2_sparsity.jpg")

    tmp_w_train_lmb = w_train_lmb.copy()
    top_5_features = {}

    for lamb in _lambdas:
        i = 0
        top_5_features[str(lamb)] = {}
        while i < 5:
            idx_max = tmp_w_train_lmb[str(lamb)].idxmax()
            top_5_features[str(lamb)][idx_max[0]] = float(tmp_w_train_lmb[str(lamb)].loc[idx_max[0]])
            tmp_w_train_lmb[str(lamb)] = tmp_w_train_lmb[str(lamb)].drop(idx_max[0])
            i += 1

    for lamb in _lambdas:
        print("Top 5 features of model trained by lambda = %s:" % str(lamb))
        print(top_5_features[str(lamb)])
        print()

    print("\n---------- PART II: Training and experimenting with IA2-train-noisy data ----------\n")
    print("\n---------- PREPROCESSING ----------\n")
    df_train_noisy = pd.read_csv("IA2-train-noisy.csv")

    df_norm_train_noisy = df_train_noisy
    df_norm_val_noisy = df_val.copy()

    for col in numerical_feas:
        mu = df_norm_train_noisy[col].mean()
        sigma = df_norm_train_noisy[col].std()

        df_norm_train_noisy[col] = normalize(df_train_noisy, col, mu, sigma)
        df_norm_val_noisy[col] = normalize(df_val, col, mu, sigma)

    print("\n---------- TRAINING ----------\n")
    acc_noisy_train_lmb = {} # dictionary to store train acc at each iteration for all models, indexed by lambda
    acc_noisy_val_lmb = {} # dictionary to store val acc at each iteration for all models, indexed by lambda
    w_noisy_train_lmb = {} # dictionary to store weights for all models, indexed by learning rate

    for lmbd, lr in zip(_lambdas, lrs):
        w,acc_train,acc_val = LR_L2_train(df_norm_train_noisy, df_norm_val_noisy, lmbd, lr, True) # train model with current lambda value
        acc_noisy_train_lmb[str(lmbd)] = acc_train # store training accuracy, indexed by lambda value
        acc_noisy_val_lmb[str(lmbd)] = acc_val # store val accuracy, indexed by lambda value
        w_noisy_train_lmb[str(lmbd)] = w # store final training weights, indexed by lambda value

    plot_losses(acc_noisy_train_lmb, acc_noisy_val_lmb, "L2_noise_accuracy_per_iteration.jpg")
    plot_acc_per_lambda(acc_noisy_train_lmb, acc_noisy_val_lmb, "L2_noise_accuracy_per_lambda.jpg")
    sparsity_graph(w_noisy_train_lmb, "L2_noise_sparsity.jpg")

    print("\n---------- PART III: Logistic regression with L1 regularization and lamba experiments ----------\n")
    print("\n---------- TRAINING ----------\n")
    _lambdas = [10**(x) for x in range(-4,3)] # generate values of lambda 10^i, i in [-4,2]
    lrs = [0.01, 0.03, 0.02, 0.08, 0.08, 0.5, 0.0001] # Learning rates that work best for corresponding lambda

    train_acc_lmb = {} # dictionary to store train acc at each iteration for all models, indexed by lambda
    val_acc_lmb = {} # dictionary to store val acc at each iteration for all models, indexed by lambda
    train_w_lmb = {} # dictionary to store weights for all models, indexed by learning rate

    for lmbd, lr in zip(_lambdas, lrs): # Iterate over lambdas and correlated learning rates
        tmpW, tmpTA, tmpVA = LR_L1_train(df_norm_train, df_norm_val, lmbd, lr) # train model with current lambda value
        train_acc_lmb[str(lmbd)] = tmpTA # store training accuracy, indexed by lambda value
        val_acc_lmb[str(lmbd)] = tmpVA # store val accuracy, indexed by lambda value
        train_w_lmb[str(lmbd)] = tmpW # store final training weights, indexed by lambda value
        
    plot_losses(train_acc_lmb, val_acc_lmb, "L1_accuracy_per_iteration.jpg") # plot accuracy train & val
    plot_acc_per_lambda(train_acc_lmb, val_acc_lmb, "L1_accuracy_per_lambda.jpg")
    sparsity_graph(train_w_lmb, "L1_sparsity.jpg") # plot sparsity
    


if __name__ == "__main__":
    main()
