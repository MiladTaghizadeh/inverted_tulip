import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm


'''
This code uploads a matrix with .mat file format and does weighted multiple quadratic 
regression. The data in the input matrix should be as follows:
    
    y1 x11 x12 x13 ... x1n
    y2 x21 x22 x33 ... x2n
    y3 ...
    .
    .
    .
    yn ...
    
Milad Taghizadeh 08/29/2024

'''

# Load the .mat file
mat = scipy.io.loadmat('highest_entropy_5000_samples.mat')

# Extract the matrix
data = mat['data']
data = np.delete(data, 1, axis=1) # delete the v* data


# Set a random seed for reproducibility
np.random.seed(27)

# a function to split data and choose samples randomly for test and train
def split_matrix(matrix, test_fraction=0.2):
    # Get the number of rows in the matrix
    num_rows = matrix.shape[0]
    
    # Calculate the number of test rows
    num_test_rows = int(num_rows * test_fraction)
    
    # Randomly select indices for the test set
    test_indices = np.random.choice(num_rows, num_test_rows, replace=False)
    
    # Create test and training sets based on the indices
    test_matrix = matrix[test_indices, :]
    train_indices = np.setdiff1d(np.arange(num_rows), test_indices)
    train_matrix = matrix[train_indices, :]
    
    return train_matrix, test_matrix

# splitting the data
train_matrix, test_matrix = split_matrix(data)

## add a row of zeros
# Row of zeros
zero_row = np.zeros(train_matrix.shape[1])

# Adding the row of zeros
train_matrix = np.vstack([zero_row, train_matrix])


# Split the training and testing to separate independent variable
training_P_star = train_matrix[:, 0]        
training_inputs = train_matrix[:, 1:]

testing_P_star = test_matrix[:, 0]
testing_inputs  = test_matrix[:, 1:]

## a function to transform input for quadratic regression

def transform_matrix(input_matrix):
    # Initialize the output matrix with zeros
    output_matrix = np.zeros((input_matrix.shape[0], int((2*training_inputs.shape[1]+training_inputs.shape[1]*(training_inputs.shape[1]-1)/2))))
    
    # Copy the first columns from the input matrix
    output_matrix[:, 0:training_inputs.shape[1]] = input_matrix
    
    # Fill columns with the squared values of input matrix
    output_matrix[:, training_inputs.shape[1]:2*training_inputs.shape[1]] = input_matrix ** 2
    
    # Fill the rest with the products of different column pairs
    col_index = 2*training_inputs.shape[1]
    for i in range(training_inputs.shape[1]):
        for j in range(i + 1, training_inputs.shape[1]):
            output_matrix[:, col_index] = input_matrix[:, i] * input_matrix[:, j]
            col_index += 1
    
    return output_matrix

def weighted_multiple_regression(X, y, weights):
 
    # Add a constant (intercept) to the model
    X_with_intercept = sm.add_constant(X)

    # Create a weighted least squares model
    model = sm.WLS(y, X_with_intercept, weights=weights)
    
    # Fit the model
    results = model.fit()

    return results


## form the weight vector
data_number = 0;
data_wieght = 5000;
weights = np.ones(training_inputs.shape[0])  
weights[data_number] = data_wieght

# Perform weighted multiple regression
results = weighted_multiple_regression(transform_matrix(training_inputs), training_P_star, weights)

## display coefficients
a = results.params
c = results.params[1:]
c0 = results.params[0]

## Gradient and Hessian matrices
G = c[:training_inputs.shape[1]]
H = np.zeros((training_inputs.shape[1],training_inputs.shape[1]))

for k in range(training_inputs.shape[1]):
    H[k,k] = c[k+training_inputs.shape[1]]
    
count = 0;
    
for l in range(0,training_inputs.shape[1]):
    for m in range(l + 1, training_inputs.shape[1]):
        H[l,m] = H[m,l] = c[2*training_inputs.shape[1]+count]
        count = count + 1

H = 2*H



## for single test
# x = np.array([[0.4, 0.1, 1, -1.5, 0, 0, 0, 0]])
# x_transformed = transform_matrix(x)
# Single_P_star_pred = regressor.predict(x_transformed)

## predictions
# Add an intercept to the new input data for prediction
testing_inputs_transformed = transform_matrix(testing_inputs)
add_intercept = sm.add_constant(testing_inputs_transformed)
P_star_pred = results.predict(add_intercept)

## RMSE calculator
mean_square_error = np.square(np.subtract(testing_P_star,P_star_pred)).mean()  
root_mean_square_error = math.sqrt(mean_square_error)

## Compute RRMSE
rrmse = root_mean_square_error / np.mean(testing_P_star)

######################### making prediction ##################################
nmesh = 15
npar  = 8

test1 = np.zeros((nmesh*nmesh,npar))

DG_AB     = np.linspace(0, 0.5, nmesh)
DG_AC     = 0
omg_AB_b  = np.linspace(-1.5, 1.5, nmesh)
omg_AC_b  = 0
omg_BC_b  = 0
omg_AB_gb = 0
omg_AC_gb = 0
omg_BC_gb = 0

# modify this for the rest of the parameters
test1[:, 1] = DG_AC*np.ones(nmesh*nmesh)
test1[:, 3] = omg_AC_b*np.ones(nmesh*nmesh)
test1[:, 4] = omg_BC_b*np.ones(nmesh*nmesh)
test1[:, 5] = omg_AB_gb*np.ones(nmesh*nmesh)
test1[:, 6] = omg_AC_gb*np.ones(nmesh*nmesh)
test1[:, 7] = omg_BC_gb*np.ones(nmesh*nmesh)


# Create an empty list to store the combinations
combinations1 = []
pars = ['DG_AB','DG_AC','omg_AB_b','omg_AC_b','omg_BC_b','omg_AB_gb','omg_AC_gb','omg_BC_gb']

## graph parameters to be plotted
par1g = DG_AB
par2g = omg_AB_b
par1gc = 'DG_AB'
par2gc = 'omg_AB_b'

# Use nested loops to combine the vectors
for i in par1g:
    for j in par2g:
        combinations1.append([i, j])
        

# Convert the list to a NumPy array with shape (25, 2)
matrix1 = np.array(combinations1)


# Place the 25x2 matrix into the 25x8 matrix
test1[:, pars.index(par1gc)] = matrix1[:,0]
test1[:, pars.index(par2gc)] = matrix1[:,1]




test1_transformed = transform_matrix(test1)
add_intercept_test1 = sm.add_constant(test1_transformed, has_constant='add')
P_star_pred_test1 = results.predict(add_intercept_test1)


X1, Y1 = np.meshgrid(par1g, par2g)
Z1 = P_star_pred_test1.reshape(nmesh, nmesh)

###################### Graph1 ############################################
# Define figure size
fig = plt.figure(figsize=(5, 5))  # Adjust width and height as needed

# Scatter plot
plt.scatter(testing_P_star, P_star_pred, color='crimson')

# Set axis limits to start from 0.00
plt.xlim(0.00, 0.3)
plt.ylim(0.00, 0.3)

# Set exactly 4 equispaced ticks on both axes
plt.xticks(np.round(np.linspace(0.00, 0.3, 4), 2))
plt.yticks(np.round(np.linspace(0.00, 0.3, 4), 2))

# Labels and title
plt.title(f'RRMSE = {rrmse:.2g}', fontsize=18, fontname='Arial')
plt.xlabel(r'Ground-truth $\bar{P}^*$', fontsize=18, fontname='Arial')
plt.ylabel('Predictions', fontsize=18, fontname='Arial')

# Aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# Unity line
plt.plot(np.linspace(0, 0.3, 100), np.linspace(0, 0.3, 100), 
         label='Unity Line (y = x)', color='black', linestyle='--')

# Tick size
plt.tick_params(axis='both', labelsize=18)
plt.tight_layout()  # Automatically adjust padding


# Show plot
plt.show()

