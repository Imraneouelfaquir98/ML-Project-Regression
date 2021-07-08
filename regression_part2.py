import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Loading Dataset
url = "Part2_data12.csv"
dataset = pd.read_csv(url)

#Shape : Dimensions of the dataset
print("Dimension = {0:d}*{1:d}".format(dataset.shape[0],dataset.shape[1]))

#Randomly mix the dataset by the function simple

dataframe = pd.DataFrame(dataset)
dataframe_mixed = dataframe.sample(frac=1)
dataframe_mixed

#We take 70% as a training data and 30% as a testing data
training_dataset_df = dataframe_mixed[:int(len(dataframe)*0.7)]
training_dataset

testing_dataset_df  = dataframe_mixed[int(len(dataframe)*0.7):len(dataframe)]
testing_dataset

# Saving training and testing dataset to a cqsv files
training_dataset_df.to_csv(r'training_dataset_part2.csv', index=False, header=True)
testing_dataset_df.to_csv(r'testing_dataset_part2.csv'  , index=False, header=True)

#Read and visualize the given data.

training_dataset_url = "training_dataset_part2.csv"
testing_dataset_url  = "testing_dataset_part2.csv"

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

# Use scatter plot to represent your training dataset.

# Plotting the training dataset
plt.scatter(training_dataset['x'], training_dataset['y'], c = 'red', marker = '*')

# Question2: Build the Polynomial regression algorithm of degree k ≥ 2
# The hypothesis function. 
def h(x,w):
    polynom = float(0)
    for i in range(len(w)):
        polynom = polynom + w[i]*pow(x,i)
    return polynom

# The empirical error MSE
def MSE(w,d_set):
    MSE = 0
    for i in range(len(d_set)):
        MSE = MSE + pow((d_set['y'][i] - h(d_set['x'][i],w)),2)
    return MSE/len(d_set)

# Calculation of the step Armijo
def armijo(wk, dk, d_set, empiricalError = MSE):
    alpha = 1
    X = 0.25*alpha*np.matmul(gradient(wk, d_set,empiricalError), dk)
    while empiricalError(np.add(wk, np.multiply(alpha,dk)), d_set) - empiricalError(wk, d_set) > X:
        alpha = alpha/2
        X = 0.25*alpha*np.matmul(gradient(wk, d_set,empiricalError), dk)
    return alpha

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

# Plotting the training dataset with hypotesis function
def plot_h(w,df):
    plt.scatter(df['x'], df['y'], c = 'red', marker = '*')
    x   = []
    h_x = []
    k = int(df['x'].min())-1
    while int(df['x'].max())+2>k : 
        x.append(k)
        h_x.append(h(k,w))
        k = k + 0.5
    plt.plot(x, h_x, linewidth=4)

# Gradient descent method
def gradientDescent(delta, degree, d_set, armij=False,alpha=0,empiricalError=MSE):
    wk = [float(0) for i in range(degree+1)]
    gk = np.multiply(-1,gradient(wk, d_set, empiricalError))
    k = 0
    while np.linalg.norm(gk)>delta:
        if k>500:
            break
        print("k = {0:d}, norm = {1:.2f}, MSE = {1:.2f}".format(k,np.linalg.norm(gk),empiricalError(wk,d_set)))
        if armij == True:
            alpha = armijo(wk,gk, d_set)
        wk = np.add(wk, np.multiply(alpha,gk))
        gk = np.multiply(-1,gradient(wk, d_set, empiricalError))
        k = k+1
    return wk

# Calculation of polynomial regression with degree 2
degree = 2
# Using alpha = 0.02 as a learning rate
w_optim_002    = gradientDescent(0.1, degree, training_dataset, alpha  = 0.02)
# Armijo's learning rate
w_optim_armijo = gradientDescent(0.1, degree, training_dataset, armij = True)

# alpha  = 0.02
plot_h(w_optim_002,training_dataset)
# armijo
plot_h(w_optim_armijo,training_dataset)

# Question6: Now, evaluate your models with the testing dataset and return the generalization error. 

# Calculation of the general error for our 4 models
# the model with alpha = 0.02
MSE_alpha002 = MSE(w_optim_002   , testing_dataset)
print("MSE = {0:.2f}".format(MSE_alpha002))
# the model with armijo learning rate
MSE_armijo   = MSE(w_optim_armijo, testing_dataset)
print("MSE = {0:.2f}".format(MSE_armijo))

# Bias
def bias(w,d_set):
    Bias = 0
    for i in range(len(d_set)):
        Bias = Bias + pow(h(d_set['x'][i],w) - d_set['y'][i],2)
    return np.sqrt(Bias)
# Variance
def variance(w,d_set):
    mean_Ypred = 0
    for i in range(len(d_set)):
        mean_Ypred = mean_Ypred + h(d_set['x'][i],w)
    mean_Ypred = mean_Ypred/len(d_set)
    Var = 0
    for i in range(len(d_set)):
        Var = Var + pow(h(d_set['x'][i],w) - mean_Ypred,2)
    return Var/len(d_set)

print("model 1 Bias = {0:.2f}".format(bias(w_optim_002,dataframe)))
print("model 2 Bias = {0:.2f}".format(bias(w_optim_armijo,dataframe)))

print("model 1 Variance = {0:.2f}".format(variance(w_optim_002,dataframe)))
print("model 2 Variance = {0:.2f}".format(variance(w_optim_armijo,dataframe)))

# Calculation of polynomial regression with degree 2
degree = 3
# Using alpha = 0.02 as a learning rate
w_optim_002_3    = gradientDescent(0.1, degree, training_dataset, alpha  = 0.02) 
# Armijo's learning rate
w_optim_armijo_3 = gradientDescent(0.1, degree, training_dataset, armij = True)

# Calculation of the general error for our 2 models
# the model with alpha = 0.02
MSE_alpha002_3 = MSE(w_optim_002_3   , testing_dataset)
print("MSE = {0:.2f}".format(MSE_alpha002))
# the model with armijo learning rate
MSE_armijo_3   = MSE(w_optim_armijo_3, testing_dataset)
print("MSE = {0:.2f}".format(MSE_armijo))

print("model 1 Bias = {0:.2f}".format(bias(w_optim_002_3,dataframe)))
print("model 2 Bias = {0:.2f}".format(bias(w_optim_armijo_3,dataframe)))

print("model 1 Variance = {0:.2f}".format(variance(w_optim_002_3,dataframe)))
print("model 2 Variance = {0:.2f}".format(variance(w_optim_armijo_3,dataframe)))

# Calculation of polynomial regression with degree 2
degree = 4
# Using alpha = 0.02 as a learning rate
w_optim_002_4    = gradientDescent(0.1, degree, training_dataset, alpha  = 0.02) 
# Armijo's learning rate
w_optim_armijo_4 = gradientDescent(0.1, degree, training_dataset, armij = True)

# Calculation of the general error for our 4 models
# the model with alpha = 0.02
MSE_alpha002_4 = MSE(w_optim_002_4   , testing_dataset)
print("MSE = {0:.2f}".format(MSE_alpha002_4))
# the model with armijo learning rate
MSE_armijo_4   = MSE(w_optim_armijo_4, testing_dataset)
print("MSE = {0:.2f}".format(MSE_armijo_4))

print("model 1 Bias = {0:.2f}".format(bias(w_optim_002_4,dataframe)))
print("model 2 Bias = {0:.2f}".format(bias(w_optim_armijo_4,dataframe)))

print("model 1 Variance = {0:.2f}".format(variance(w_optim_002_4,dataframe)))
print("model 2 Variance = {0:.2f}".format(variance(w_optim_armijo_4,dataframe)))

plot_h(w_optim_002_4,training_dataset)

plot_h(w_optim_armijo_4,training_dataset)

# Evaluation of polynomial model with degree 2
MSE_armijo   = MSE(w_optim_armijo, testing_dataset)
print("MSE      = {0:.2f}".format(MSE_armijo))
print("Bias     = {0:.2f}".format(bias(w_optim_armijo,dataframe)))
print("Variance = {0:.2f}".format(variance(w_optim_armijo,dataframe)))

# Evaluation of polynomial model with degree 3
MSE_armijo_3   = MSE(w_optim_armijo_3, testing_dataset)
print("MSE      = {0:.2f}".format(MSE_armijo_3))
print("Bias     = {0:.2f}".format(bias(w_optim_armijo_3,dataframe)))
print("Variance = {0:.2f}".format(variance(w_optim_armijo_3,dataframe)))

# Evaluation of polynomial model with degree 4
MSE_armijo_4   = MSE(w_optim_armijo_4, testing_dataset)
print("MSE      = {0:.2f}".format(MSE_armijo_4))
print("Bias     = {0:.2f}".format(bias(w_optim_armijo_4,dataframe)))
print("Variance = {0:.2f}".format(variance(w_optim_armijo_4,dataframe)))











