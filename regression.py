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


training_dataset_df = dataframe_mixed[:int(len(dataframe)*0.7)]
training_dataset

testing_dataset_df  = dataframe_mixed[int(len(dataframe)*0.7):len(dataframe)]
testing_dataset

# Saving training and testing dataset to a cqsv files
training_dataset_df.to_csv(r'training_dataset.csv', index=False, header=True)
testing_dataset_df.to_csv(r'testing_dataset.csv'  , index=False, header=True)


training_dataset_url = "training_dataset.csv"
testing_dataset_url  = "testing_dataset.csv"

training_dataset = pd.read_csv(training_dataset_url)
testing_dataset  = pd.read_csv(testing_dataset_url)

# Give the mean/median/min/max for each feature.
# Calculation of the mean of x
print("mean(x) = {:.2f}".format(float(pd.DataFrame(dataset['x']).mean())))

# Calculation of the median of x
print("median(x) = {:.2f}".format(float(pd.DataFrame(dataset['x']).median())))

# Calculation of the max of x
print("max(x) = {:.2f}".format(float(pd.DataFrame(dataset['x']).max())))

# Calculation of the min of x
print("min(x) = {:.2f}".format(float(pd.DataFrame(dataset['x']).min())))

# Question2: Use scatter plot to represent your training dataset.

# Plotting the training dataset
plt.scatter(training_dataset['x'], training_dataset['y'], c = 'red', marker = '*')

# Plotting the training dataset with hypotesis function
def plot_h(w,df):
    plt.scatter(df['x'], df['y'], c = 'red', marker = '*')
    plt.plot([df['x'].min(),df['x'].max()],[h(df['x'].min(),w),h(df['x'].max(),w)], linewidth=4)

# Question5:

# The hypothesis function. 

# The hypothesis function. 
def h(x,w):
    return w[1]*x + w[0]

# The empirical error
# The empirical error MSE
def MSE(w,d_set):
    MSE = 0
    for i in range(len(d_set)):
        MSE = MSE + pow((d_set['y'][i] - h(d_set['x'][i],w)),2)
    return MSE/len(d_set)

# The empirical error RMSE
def RMSE(w,d_set):
    return np.sqrt(MSE(w,d_set))

# The empirical error MAE
def MAE(w,d_set):
    MAE = float(0)
    for i in range(len(d_set)):
        MAE = MAE + np.abs((d_set['y'][i] - h(d_set['x'][i],w)))
    return MAE/len(d_set)

# Calculation of the gradient
def gradient(w, d_set, empiricalError = MSE):
    grad = []
    wi = [float(0) for i in range(len(w))]
    for i in range(len(w)):
        for k in range(len(w)):
            wi[k] = w[k]
        wi[i] = wi[i] + 1e-10
        grad.append((empiricalError(wi,d_set) - empiricalError(w,d_set))/1e-10)
    return grad

# Calculation of the step Armijo
def armijo(wk, dk, d_set, empiricalError = MSE):
    alpha = 1
    X = 0.25*alpha*np.matmul(gradient(wk, d_set,empiricalError), dk)
    while empiricalError(np.add(wk, np.multiply(alpha,dk)), d_set) - empiricalError(wk, d_set) > X:
        alpha = alpha/2
        X = 0.25*alpha*np.matmul(gradient(wk, d_set,empiricalError), dk)
    return alpha

# Gradient descent method
def gradientDescent(delta, wk, d_set, armij = False, alpha = 0, empiricalError = MSE):
    gk = np.multiply(-1,gradient(wk, d_set, empiricalError))
    k = 0
    while k<500:
        if armij == True:
            alpha = armijo(wk,gk, d_set)
        wk = np.add(wk, np.multiply(alpha,gk))
        gk = np.multiply(-1,gradient(wk, d_set, empiricalError))
        k = k+1
    return wk

# Calculation of the regression model for our four model

w_optim_001 = gradientDescent(0.1, [1,1], training_dataset, alpha  = 0.01) # MSE
print("alpha=0.01: w* = [{0:.2f}, {1:.2f}], Emp_Error = {2:.2f} ".format(w_optim_001[0],w_optim_001[1],MSE(w_optim_001,training_dataset)))

w_optim_002 = gradientDescent(0.1, [1,1], training_dataset, alpha  = 0.02) # RMSE
print("alpha=0.02: w* = [{0:.2f}, {1:.2f}], Emp_Error = {2:.2f} ".format(w_optim_002[0],w_optim_002[1],MSE(w_optim_002,training_dataset)))

w_optim_003 = gradientDescent(0.1, [1,1], training_dataset, alpha  = 0.03) # RMSE
print("alpha=0.03: w* = [{0:.2f}, {1:.2f}], Emp_Error = {2:.2f} ".format(w_optim_003[0],w_optim_003[1],MSE(w_optim_003,training_dataset)))

w_optim_armijo  = gradientDescent(0.1, [1,1], training_dataset, armij = True) # MAE
print("Armijo    : w* = [{0:.2f}, {1:.2f}], Emp_Error = {2:.2f} ".format(w_optim_armijo[0],w_optim_armijo[1],MSE(w_optim_armijo,training_dataset)))


# alpha  = 0.01
plot_h(w_optim_001,training_dataset)

# alpha  = 0.02
plot_h(w_optim_002,training_dataset)

# alpha  = 0.03
plot_h(w_optim_003,training_dataset)

# Armijo step
plot_h(w_optim_armijo,training_dataset)

#Question7: Evaluation of our 4 models with the testing dataset and return the generalization error.

# Calculation of the general error for our 4 models
# the model with alpha = 0.01
MSE_alpha001 = MSE(w_optim_001   , testing_dataset)
print("MSE = {0:.2f}".format(MSE_alpha001))
# the model with alpha = 0.02
MSE_alpha002 = MSE(w_optim_002   , testing_dataset)
print("MSE = {0:.2f}".format(MSE_alpha002))
# the model with alpha = 0.03
MSE_alpha003 = MSE(w_optim_003   , testing_dataset)
print("MSE = {0:.2f}".format(MSE_alpha003))
# the model with armijo learning rate
MSE_armijo   = MSE(w_optim_armijo, testing_dataset)
print("MSE = {0:.2f}".format(MSE_armijo))

#Question8: Estimate generalization error (bias-variance) and interpret results
def bias(w,d_set):
    Bias = 0
    for i in range(len(d_set)):
        Bias = Bias + pow(h(d_set['x'][i],w) - d_set['y'][i],2)
    return np.sqrt(Bias)

print("model 1 Bias = {0:.2f}".format(bias(w_optim_001,dataframe)))
print("model 2 Bias = {0:.2f}".format(bias(w_optim_002,dataframe)))
print("model 3 Bias = {0:.2f}".format(bias(w_optim_003,dataframe)))
print("model 4 Bias = {0:.2f}".format(bias(w_optim_armijo,dataframe)))

def variance(w,d_set):
    mean_Ypred = 0
    for i in range(len(d_set)):
        mean_Ypred = mean_Ypred + h(d_set['x'][i],w)
    mean_Ypred = mean_Ypred/len(d_set)
    
    Var = 0
    for i in range(len(d_set)):
        Var = Var + pow(h(d_set['x'][i],w) - mean_Ypred,2)
    return Var/len(d_set)

print("model 1 Variance = {0:.2f}".format(variance(w_optim_001,dataframe)))
print("model 2 Variance = {0:.2f}".format(variance(w_optim_002,dataframe)))
print("model 3 Variance = {0:.2f}".format(variance(w_optim_003,dataframe)))
print("model 4 Variance = {0:.2f}".format(variance(w_optim_armijo,dataframe)))



