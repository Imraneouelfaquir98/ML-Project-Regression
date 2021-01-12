import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Loading Dataset
url = "Part1_data12.csv"
dataset = pd.read_csv(url)

#Shape : Dimensions of the dataset
print("Dimension = {0:d}*{1:d}".format(dataset.shape[0],dataset.shape[1]))

dataframe = pd.DataFrame(dataset)
dataframe_mixed = dataframe.sample(frac=1)
# dataframe_mixed


# training_dataset
training_dataset = dataframe_mixed[:int(len(dataframe)*0.7)]

# testing_dataset
testing_dataset  = dataframe_mixed[int(len(dataframe)*0.7):len(dataframe)]

# Saving training and testing dataset to a cqsv files
training_dataset.to_csv(r'training_dataset.csv', index=False, header=True)
testing_dataset.to_csv(r'testing_dataset.csv'  , index=False, header=True)

# Read and visualize the given data.
training_dataset_url = "training_dataset.csv"
testing_dataset_url  = "testing_dataset.csv"

training_dataset = pd.read_csv(training_dataset_url)
testing_dataset  = pd.read_csv(testing_dataset_url)

# Calculation of the mean of x
print("mean(x) = {:.2f}".format(int(pd.DataFrame(dataset['x']).mean())))

# Calculation of the median of x
print("median(x) = {:.2f}".format(int(pd.DataFrame(dataset['x']).median())))

# Calculation of the max of x
print("max(x) = {:.2f}".format(int(pd.DataFrame(dataset['x']).max())))

# Calculation of the min of x
print("min(x) = {:.2f}".format(int(pd.DataFrame(dataset['x']).min())))

# Plotting the training dataset
plt.scatter(training_dataset['x'], training_dataset['y'], c = 'red', marker = '*')

# Function for lotting the training dataset with hypotesis function
def plot_h(w):
    data = pd.DataFrame(trainig_dataset['x'])
    plt.scatter(training_dataset['x'], training_dataset['y'], c = 'red', marker = '*')
    plt.plot([data.min(),data.max()],[h(data.min(),w),h(data.max(),w)], linewidth=4)

# The hypothesis function. 
def h(x,w):
    return w[1]*x + w[0]

# The empirical error MSE
def MSE(w):
    MSE = 0
    for i in range(len(training_dataset)):
        MSE = MSE + pow((training_dataset['y'][i] - h(training_dataset['x'][i],w)),2)
    return MSE/len(training_dataset)

# The empirical error RMSE
def RMSE(w):
    return np.sqrt(MSE(w))

# The empirical error MAE
def MAE(w):
    MAE = float(0)
    for i in range(len(training_dataset)):
        MAE = MAE + np.abs((training_dataset['y'][i] - h(training_dataset['x'][i],w)))
    return MAE/len(training_dataset)

# Calculation of the gradient
def gradient(w, empiricalError = MSE):
    grad = []
    wi = [0.0000, 0.0000]
    for i in range(2):
        for k in range(2):
            wi[k] = w[k]
        wi[i] = wi[i] + 1e-10
        grad.append((empiricalError(wi) - empiricalError(w))/1e-10)
    return grad

# Calculation of the step Armijo
def armijo(wk, dk, empiricalError = MSE):
    alpha = 1
    X = 0.25*alpha*np.matmul(gradient(wk,empiricalError), dk)
    while empiricalError(np.add(wk, np.multiply(alpha,dk))) - empiricalError(wk) > X:
        alpha = alpha/2
        X = 0.25*alpha*np.matmul(gradient(wk,empiricalError), dk)
    return alpha

# Gradient descent method
def gradientDescent(delta, wk, empiricalError = MSE):
    gk = np.multiply(-1,gradient(wk))
    alpha = 0
    k = 0
    while k<10:
        alpha = armijo(wk,gk, empiricalError)
        wk = np.add(wk, np.multiply(alpha,gk))
        gk = np.multiply(-1,gradient(wk, empiricalError))
        k = k+1
    return wk
	# print("{0:d}- Ls = {1:.2f}".format(int(k),float(empiricalError(wk))))

# Calculation of the regression model for each empirical error
w_optim_MSE  = gradientDescent(0.1, [1,1],  MSE)
w_optim_RMSE = gradientDescent(0.1, [1,1], RMSE)
w_optim_MAE  = gradientDescent(0.1, [1,1],  MAE)

# MAE
print(w_optim_MAE)
plot_h(w_optim_MAE)

# MSE
print(w_optim_MSE)
plot_h(w_optim_MSE)

# RMSE
print(w_optim_RMSE)
plot_h(w_optim_RMSE)