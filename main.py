# Import
import numpy as np
import matplotlib.pyplot as plt
import random

# Define function to generate mutivariate gaussian samples
def generate_multivariate_samples(mean, covariance_matrix, num_samples):
    samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)
    return samples

# Define the mean vector and covariance matrix
mean_vector = np.array([0, 0])
covariance_matrix = np.array([[1, 0.5], [0.5, 2]])

# Number of samples to generate
num_samples = 1000

# Generate multivariate Gaussian samples
samples = generate_multivariate_samples(mean_vector, covariance_matrix, num_samples)

print("Generated samples shape:", samples.shape)
print("First 5 samples:")
print(samples[:5])


# Generating synthetic dataset
N = 1000
mean1 = np.array([0,0])
mean2 = np.array([4,4])
cov1 = np.array([[1,0], [0,1]])
cov2 = np.array([[3,0], [0,3]])
X = np.zeros([N, 2])
y = np.zeros(N)
for i in range(N):
    label = np.random.choice([0, 1], p = [0.5, 0.5])
    y[i] = label
    if(label == 0):
        X[i] = np.random.multivariate_normal(mean1, cov1, 1)
    else:
        X[i] = np.random.multivariate_normal(mean2, cov2, 1)


# Define function to plot samples
def plot_samples(X, y, title):
    # Extract x and y coordinates from the samples
    X1 = []
    X2 = []
    
    for i in range(X.shape[0]):
        if(y[i] == 0):
            X1.append(X[i])
        else:
            X2.append(X[i])
            
    X1 = np.array(X1)
    X2 = np.array(X2)

    # Plot the samples
    plt.figure(figsize=(8, 6))
    plt.scatter(X1[:, 0], X1[:, 1], alpha=0.5, color = 'red', label = 'class 1')
    plt.scatter(X2[:, 0], X2[:, 1], alpha=0.5, color = 'blue', label = 'class 2')
    
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.legend()
    plt.show()
    

# Plot the training samples
plot_samples(X, y, "Training Dataset")

# Define the signmoid function
def sigmoid(t):
    return np.exp(t)/(1 + np.exp(t))

# Define function to compute mu( = sigmoid(x'w)) values of the data points
def compute_mu(X, w):
    N = X.shape[0]
    mu = np.zeros(N)
    for i in range(N):
        mu[i] = np.dot(w, X[i].reshape(2, ))
        mu[i] = sigmoid(mu[i])
        
    return mu


# Define function to compute the MAP estimate of w assuming a gaussian prior as N(w| 0, (1/alpha)I)
def MAP(alpha, X, y):
    N, D = X.shape
    w_MAP = np.zeros(D)
    limit = 1e-9
    step_size = 0.001
    cnt = 0
    while(True):
        cnt += 1
        z = np.dot(X, w_MAP)
        h = sigmoid(z)
        
        gradient = np.dot(X.T, (h - y)) + alpha * w_MAP
        if(np.linalg.norm(step_size * gradient) < limit):
            break
        
        w_MAP -= step_size * gradient
        
    return w_MAP, cnt


# Define the function to compute HESSIAN matrix of the negative log posterior of w( = -log(w|X, y))
def hessian(X, w_MAP, alpha):
    N, D = X.shape
    mu = compute_mu(X, w_MAP)
    S = np.zeros([N, N])
    for i in range(N):
        S[i][i] = mu[i] * (1 - mu[i])
    
    hessian = np.matmul(X.T, np.matmul(S, X)) + alpha * np.eye(D)
    return hessian


# Computing the Gaussian posterior parameters(Laplace approximation)
alpha = 1
w_MAP, cnt = MAP(alpha, X, y)
cov = np.linalg.inv(hessian(X, w_MAP, alpha))

print("MAP estimate of w:")
print(w_MAP)
print('\n')
print("Covariance of the Gaussian posterior(Laplace approximation) over w:")
print(cov)
print('\n')
print("Number of Iterations:", cnt)


# Define function to compute the gradient of the prediction-regularized log joint posterior
# x: Query point
# lamda: regularization constant
def grad(X, y, w, x, lamda, alpha):
    mu = compute_mu(X, w)
    return np.matmul(X.T, y - mu) - alpha * w + lamda * x


# Define function to compute the prediction-regularized MAP estimate by maximizing the prediction-regularized log joint posterior
def PR_MAP(X, y, lamda, x, w_MAP, alpha):
    N, D = X.shape
    limit = 1e-12
    w_PR_MAP = w_MAP + 1e-9 * np.ones(2)     # initialized wvery close to the MAP estimate
    step_size = 1e-3
    cnt = 0
    while(True):
        gradient = grad(X, y, w_PR_MAP, x, lamda, alpha)
        if(np.linalg.norm(step_size * gradient) < limit):
            break
            
        w_PR_MAP += (step_size * gradient)
        cnt += 1
    
    return w_PR_MAP, cnt


# Let's take a query point
q = np.array([1,2])


# Computing the PR_MAP for the query point
lamda = 1e-6     # network regularization condstant
w_PR_MAP, cnt = PR_MAP(X, y, lamda, q, w_MAP, alpha)
print("PR_MAP estimate of w:")
print(w_PR_MAP)
print('\n')
print("Number of iteratiions: ", cnt)
# Computing the PPD variance of the query point using both the models (Laplace and HFL)

# Laplace method
pred_var_laplace = np.dot(q, np.matmul(cov.T, q))
print("PPD variance of query point (Laplace)", pred_var_laplace)

# HFL method
pred_var_hfl = (1/lamda) * (np.dot(w_PR_MAP - w_MAP, q))
print("PPD variance of query point (HFL)", pred_var_hfl)

# Generating testing data
num_test = 250
X_test = np.zeros([num_test, 2])
y_test = np.zeros(num_test)
for i in range(num_test):
    label = np.random.choice([0, 1], p = [0.5, 0.5])
    y_test[i] = label
    if(label == 0):
        X_test[i] = np.random.multivariate_normal(mean1, cov1, 1)
    else:
        X_test[i] = np.random.multivariate_normal(mean2, cov2, 1)
        
# Plotting the test samples
plot_samples(X_test, y_test, "Testing Dataset")

# Computing the PPD variance(= var(w'x)) of all the testing points using Laplace method
pred_var_test_laplace = np.zeros(num_test)
for i in range(num_test):
    pred_var_test_laplace[i] = np.dot(X_test[i], np.matmul(cov.T, X_test[i]))
    
# Approximating the PPD using Monte Carlo method

# Sampling w from the posterior distribution
M = 1000
w_samples = generate_multivariate_samples(w_MAP, cov, M)

# Approximating the PPD p(y* = 1|X, y, x*)  - Laplace
ppd_laplace = np.matmul(w_samples, X_test.T)

for i in range(M):
    for j in range(num_test):
        ppd_laplace[i][j] = sigmoid(ppd_laplace[i][j])
        
ppd_laplace = np.mean(ppd_laplace, axis = 0)

# Define a function to compute PPD variance of a given query point using HFL method
def compute_ppd_var_hfl(X, y, lamda, q, w_MAP, alpha):
    w_PR_MAP,_ = PR_MAP(X, y, lamda, q, w_MAP, alpha)
    var = (1/lamda) * (np.dot(w_PR_MAP - w_MAP, q))
    return var

# Computing the PPD variance(= var(w'x)) of all testing points using HFL method
pred_var_test_hfl = np.zeros(num_test)
lamda = 1e-6
alpha = 1
for i in range(num_test):
    pred_var_test_hfl[i] = compute_ppd_var_hfl(X, y, lamda, X_test[i], w_MAP, alpha)

# Approximating the PPD p(y* = 1|X, y, x*) - HFL
pred_var_test_hfl = np.abs(pred_var_test_hfl)
ppd_hfl = np.zeros(num_test)
f_samples = np.zeros([num_test, M])
for i in range(num_test):
    f_samples[i] = np.random.normal(np.dot(w_MAP, X_test[i]), pred_var_test_hfl[i], M)

for i in range(num_test):
    for j in range(M):
        f_samples[i][j] = sigmoid(f_samples[i][j])


        
ppd_hfl = np.mean(f_samples, axis = 1)
# Computing the Binary Cross-Entropy Loss in both the cases
loss_laplace = 0
loss_hfl = 0
for i in range(num_test):
    loss_laplace += y_test[i]*np.log(ppd_laplace[i]) + (1 - y_test[i])*np.log(1 - ppd_laplace[i])
    loss_hfl += y_test[i]*np.log(ppd_hfl[i]) + (1 - y_test[i])*np.log(1 - ppd_hfl[i])

    
loss_laplace *= (-1/num_test)
loss_hfl *= (-1/num_test)
print("Binary Cross Entropy Loss (Laplace): ", loss_laplace)
print("Binary Cross Entropy Loss (HFL): ", loss_hfl)


# Assigning class to the test data as per hard classification using both methods
y_laplace = np.zeros(num_test)
y_hfl = np.zeros(num_test)
for i in range(num_test):
    if(ppd_laplace[i] > 0.5):
        y_laplace[i] = 1
    if(ppd_hfl[i] > 0.5):
        y_hfl[i] = 1

        
# Computing Accuracy of both the models as per hard classification
accuracy_laplace = 0
accuracy_hfl = 0
for i in range(num_test):
    if(y_test[i] == y_laplace[i]):
        accuracy_laplace += 1
    if(y_test[i] == y_hfl[i]):
        accuracy_hfl += 1
        
accuracy_laplace /= num_test
accuracy_hfl /= num_test

print("Laplace model accuracy: ", accuracy_laplace)
print("HFL model accuracy: ", accuracy_hfl)
# Computing number of test points where the models differ
print("Number of test points where the model differed in hard classification:", np.sum(np.abs(y_hfl - y_laplace)))