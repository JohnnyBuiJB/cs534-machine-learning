import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

def normalize(data, column, mu_dict, sigma_dict):
        data[column] = (data[column] - mu_dict[column])/sigma_dict[column]
        
        
def dataProcessing(data, mu_dict, sigma_dict):
    data["bias"] = 1
    bias = data.pop("bias")
    data.insert(0, "bias", bias)
        
    data = data.drop(columns=["id", "date", "yr_renovated"]) # drop ID column
    
    for column in data: 
        if column not in ['bias', 'waterfront', 'price']:
            normalize(data, column, mu_dict, sigma_dict)
    
    return data


def dateSplit(data):
    data['month'] = [int(x.split("/")[0]) for x in data['date'].values] # month
    data['day'] = [int(x.split("/")[1]) for x in data['date'].values] # day
    data['year'] = [int(x.split("/")[2]) for x in data['date'].values] # year
    return data


def reno(data):
    data['age_since_renovated'] = data.apply(lambda d: ((d["year"] - d["yr_built"]) if (d["yr_renovated"] == 0) 
                                                        else (d["year"] - d["yr_renovated"])), axis=1)
    return data


def main():
    train_data = pd.read_csv("IA1_train.csv")
    val_data = pd.read_csv("IA1_dev.csv")
    
    # Split date
    train_data = dateSplit(train_data)
    val_data = dateSplit(val_data)

    # Age since renovated
    train_data = dateSplit(train_data)
    val_data = dateSplit(val_data)

    # Find mu and sigma for train -> put into dict
    mu_dict = {}
    sigma_dict = {}

    for col in train_data.columns.values:
        if col not in ['id', 'date']:
            mu_dict[col] = train_data[col].mean()
            sigma_dict[col] = train_data[col].std()

            
    # Finish process data
    train_p = dataProcessing(train_data, mu_dict, sigma_dict)
    val_p = dataProcessing(val_data, mu_dict, sigma_dict)
    
    features = ["bias", "sqft_living"]

    X = train_p[features]
    y = train_p["price"].to_frame()

    X_val = val_p[features]
    y_val = val_p["price"].to_frame()
    
    learning_rate = 1

    n_iterations = 4000
    m = len(X)

    theta_init = np.random.randn(2,1)  # random initialization
    theta = theta_init

    mse_train = []
    grads_train = []
    theta_train = []

    for iter in range(n_iterations):
        # Rename the product to match y label
        y_hat = (X.dot(theta)).rename(columns={0: "price"}) - y
        
        # TODO: Early stop
        
        grads = 2/m * X.T.dot(y_hat)
        theta = theta - learning_rate * grads
        
        y_pred = (X.dot(theta)).rename(columns={0: "prices"})
        mse = float(((y - y_pred) ** 2).sum() / m)
        
        mse_train.append(mse)
        # grads_train.append(grads)
        # theta_train.append(theta)

    print(mse_train)


if __name__ == "__main__":
    main()

