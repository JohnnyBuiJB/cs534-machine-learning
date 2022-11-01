# AI534 Group 50: Vy Bui, Sebastian Mueller, Derek Helms

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 13

# Loads a data file from a provided file location.
def load_data(path):
    return pd.read_csv(path)

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    # Drop ID column
    preprocessed_data = data
    preprocessed_data = preprocessed_data.drop(columns=["id"])
    
    # Split date into month, day, and year
    preprocessed_data['month'] = [int(x.split("/")[0]) for x in preprocessed_data['date'].values] # month
    preprocessed_data['day'] = [int(x.split("/")[1]) for x in preprocessed_data['date'].values] # day
    preprocessed_data['year'] = [int(x.split("/")[2]) for x in preprocessed_data['date'].values] # year
    preprocessed_data = preprocessed_data.drop(columns=["date"])
    
    # Insert bias term (weight 0)
    preprocessed_data.insert(loc=0, column="bias", value=1)
    
    # Generate new feature based on year, year built, and year renovated
    preprocessed_data['age_since_renovated'] = preprocessed_data.apply(lambda d: ((d["year"] - d["yr_built"]) if 
                                                                                  (d["yr_renovated"] == 0) else 
                                                                                  (d["year"] - d["yr_renovated"])), axis=1)
    preprocessed_data = preprocessed_data.drop(columns=["yr_renovated"])
    
    # Generate mu and sigma for training data (also used in test data)
    if len(g_mu_dict) == 0:
        for col in preprocessed_data.columns.values:
            if col not in ['bias', 'price', 'waterfront']:
                g_mu_dict[col] = preprocessed_data[col].mean()
                g_sigma_dict[col] = preprocessed_data[col].std()
        
    # Normalize columns if flagged
    if normalize:
        for column in preprocessed_data: 
            if column not in ['bias', 'waterfront', 'price']:
                preprocessed_data[column] = ((preprocessed_data[column] - g_mu_dict[column])/g_sigma_dict[column])
    
    # Drop sqft_living 15 if flagged
    if drop_sqrt_living15:
        preprocessed_data = preprocessed_data.drop(columns=["sqft_living15"])
                
    return preprocessed_data

# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr, diverged):
    n_iterations = 4000
    n = len(data)
    n_features = len(data.columns.values)
    epsilonLow = 10**(-6) # converged values
    epsilonHigh = 10**7 # diverged value
    
    w_init = np.zeros(n_features).reshape(n_features,1)
    weights = w_init
    losses = []
    
    print("Learning Rate: ", lr) # DELETE
    for iter in range(n_iterations):
        # Rename the product to match y label
        y_hat = (data.dot(weights)).rename(columns={0: "price"})

        grads = 2/n * data.T.dot(y_hat - labels)
        weights = weights - lr * grads

        y_pred = (data.dot(weights)).rename(columns={0: "prices"})
        mse = float(((labels - y_pred) ** 2).sum() / n)
        losses.append(mse)

        magLw = np.linalg.norm(grads.values) # magnitude of gradient (change in weights)

        # Early stop due to tiny change or diverging
        if magLw < epsilonLow:
            print("Converged Early, iter: ", iter)
            diverged[str(lr)] = False
            return weights, losses, diverged

        if magLw > epsilonHigh:
            print("Diverged, iter: ", iter)
            diverged[str(lr)] = True
            return weights, losses, diverged
    
    diverged[str(lr)] = False
    
    return weights, losses, diverged

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, diverged):
    plt.figure(figsize=(14,8))
    for i, (lr, mse) in enumerate(losses.items()):
        if not diverged[lr]: # dont plot diverging models
            plt.plot(np.arange(1,len(mse)+1), mse, label=lr, linewidth=3)

    plt.legend(title="LR")   
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("MSE per Iteration by Learning Rates (Figure 1)")
    plt.savefig("MSE_per_LR.jpg")
    return


# mu and sigma from training data, used for normalization
g_mu_dict = {}
g_sigma_dict = {}

# Invoke the above functions to implement the required functionality for each part of the assignment.
def main():
    print("\n---------- PART 0: Data preprocessing ----------\n")
    train_raw = load_data("IA1_train.csv") # load train data
    val_raw = load_data("IA1_dev.csv") # load test data
    
    train_ppd_norm = preprocess_data(train_raw, 
                                normalize=True, 
                                drop_sqrt_living15=False)
    
    val_ppd_norm = preprocess_data(val_raw, 
                                normalize=True, 
                                drop_sqrt_living15=False)
    
    print("\n---------- PART 0: Done! ----------\n")

    print("\nPART 1: Implement batch gradient descent and experiment with different learning rates.\n")
    features = train_ppd_norm.columns.drop("price")
    X_norm_train = train_ppd_norm[features]
    y_train = train_ppd_norm["price"].to_frame()
    X_norm_val = val_ppd_norm[features]
    y_val = val_ppd_norm["price"].to_frame()

    norm_mse_lr = {} # dictionary to store mse at each iteration for all models, indexed by learning rate
    norm_w_lr = {} # dictionary to store weights for all models, indexed by learning rate
    norm_diverged = {}

    print("\n---------- Fitting Models ----------\n")
    learning_rates = [1, 0.178, 0.15, 0.1, 0.05, 0.01, 0.001, 0.0001]
    for lr in learning_rates:
        wghts, loss, norm_diverged = gd_train(X_norm_train, y_train, lr, norm_diverged)
        norm_mse_lr[str(lr)] = loss
        norm_w_lr[str(lr)] = wghts
        
    ### Plot MSE for each learning rate for non-diverging models (part 1a)
    plot_losses(norm_mse_lr, norm_diverged)

    ### Print Learning Rates and MSE for each non-diverging model (part 1b)
    cur_min = 1000000
    best_model = " "
    n = len(y_val)
    print("\n---------- Learning Rates and MSE ---------- \n")
    for lr, w in norm_w_lr.items():
        if not norm_diverged[lr]:
            y_pred = (X_norm_val.dot(w)).rename(columns={0: "prices"})
            mse = float(((y_val - y_pred) ** 2).sum() / n)
            print("Learning Rate:", lr + ", MSE:", mse)
            if mse < cur_min:
                cur_min = mse
                best_model = lr
                
    ### Print best model features and corresponding weights (part 1c)
    print("\n-----  Best model has a learning rate of:", best_model + ", with feature importance: -----\n")
    feat_importance = norm_w_lr[best_model].rename(columns={"price":"weights"}).drop(index="bias")
    for i, idx in enumerate(feat_importance.index.values):
        print(idx + ":", feat_importance.iloc[i].values[0])


    print("\n---------- PART 1: Done! ----------\n")
    
    print("\nPART 2 a. Training and experimenting with non-normalized data\n")    
    # Prepare data
    train_ppd_non_norm = preprocess_data(train_raw, 
                                normalize=False, 
                                drop_sqrt_living15=False)
    
    val_ppd_non_norm = preprocess_data(val_raw, 
                                normalize=False, 
                                drop_sqrt_living15=False)
    
    features = train_ppd_norm.columns.drop("price")
    X_non_norm_train = train_ppd_non_norm[features]
    y_train = train_ppd_norm["price"].to_frame()
    X_non_norm_val = val_ppd_non_norm[features]
    y_val = val_ppd_norm["price"].to_frame()
    
    print("\n---------- Fitting Models ----------\n")
    non_norm_mse_lr = {} # dictionary to store mse at each iteration for all models, indexed by learning rate
    non_norm_w_lr = {} # dictionary to store weights for all models, indexed by learning rate
    non_norm_diverged = {}
    
    learning_rates = [10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7),10**(-8),10**(-9),10**(-10),10**(-11),10**(-12)]
    for lr in learning_rates:
        wghts, loss, non_norm_diverged = gd_train(X_non_norm_train, y_train, lr, non_norm_diverged)
        non_norm_mse_lr[str(lr)] = loss
        non_norm_w_lr[str(lr)] = wghts
        
    # Use binary search to search for optimal learning rate
    max_depth = 10
    l = 10**(-11)
    r = 10**(-10)
    cur_depth = 0
    
    l_loss = non_norm_mse_lr[str(l)]
    r_loss = non_norm_mse_lr[str(r)]

    while cur_depth < max_depth:
        m = (l + r) / 2
        
        wghts, loss, non_norm_diverged = gd_train(X_non_norm_train, y_train, m, non_norm_diverged)
        non_norm_mse_lr[str(m)] = loss
        non_norm_w_lr[str(m)] = wghts
        
        if l_loss[-1] < r_loss[-1]:
            r = m
            r_loss = loss
        else:
            l = m
            l_loss = loss
        
        cur_depth += 1

    # Print Learning Rates and MSE for each non-diverging model
    print("\n---------- Learning Rates and MSE on the training data ---------- \n")
    for lr, w in non_norm_w_lr.items():
        if not non_norm_diverged[lr]:
            y_pred = (X_non_norm_train.dot(w)).rename(columns={0: "prices"})
            mse = float(((y_train - y_pred) ** 2).sum() / len(y_train))
            print("Learning Rate:", lr + ", MSE:", mse)
                
    print("\n---------- Learning Rates and MSE on the validation data ---------- \n")
    cur_min = 1000000
    best_model = " "
    for lr, w in non_norm_w_lr.items():
        if not non_norm_diverged[lr]:
            y_pred = (X_non_norm_val.dot(w)).rename(columns={0: "prices"})
            mse = float(((y_val - y_pred) ** 2).sum() / len(y_val))
            print("Learning Rate:", lr + ", MSE:", mse)
            if mse < cur_min:
                cur_min = mse
                best_model = lr
                
    # ### Print best model features and corresponding weights
    print("\n-----  Best model has a learning rate of:", best_model + ", with feature importance: -----\n")
    feat_importance = non_norm_w_lr[best_model].rename(columns={"price":"weights"}).drop(index="bias")
    for i, idx in enumerate(feat_importance.index.values):
        print(idx + ":", feat_importance.iloc[i].values[0])
        
    print("\n---------- PART 2a: Done! ----------\n")

    print("\n---------- PART 2b. Training with redundant feature removed ----------\n")    
    # Drop redundant feature
    redundant_features = ["sqft_living15"]
    X_norm_wo_r_train = X_norm_train.drop(redundant_features, axis=1)
    X_norm_wo_r_val = X_norm_val.drop(redundant_features, axis=1)

    # Train
    non_norm_wo_r_diverged = {}
    wghts, loss, non_norm_wo_r_diverged = gd_train(X_norm_wo_r_train, y_train, 
                                                   0.178, 
                                                   non_norm_wo_r_diverged)
    
    y_norm_wo_r_pred = (X_norm_wo_r_val.dot(wghts)).rename(columns={0: "prices"})
    mse = float(((y_val - y_norm_wo_r_pred) ** 2).sum() / len(y_val))

    print("MSE: ", mse)
    
    print("\n----- The feature importance: -----\n")
    feat_importance = wghts.rename(columns={"price":"weights"}).drop(index="bias")
    for i, idx in enumerate(feat_importance.index.values):
        print(idx + ":", feat_importance.iloc[i].values[0])


if __name__ == "__main__":
    main()
