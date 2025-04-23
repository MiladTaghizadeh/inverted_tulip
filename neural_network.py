import scipy.io
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import math

'''
This code uploads a matrix with .mat file format and does neural network. 
The data in the input matrix should be as follows:
    
    y1 x11 x12 x13 ... x1n
    y2 x21 x22 x33 ... x2n
    y3 ...
    .
    .
    .
    yn ...
    
Milad Taghizadeh 08/20/2024

'''

# Load the .mat file
mat = scipy.io.loadmat('highest_entropy_5000_samples.mat')

# Extract the matrix
data = mat['data']
data = np.delete(data, 1, axis=1) # delete the v* data

# Set a random seed for reproducibility
np.random.seed(42)

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


# Split the training and testing to separate independent variable
training_P_star = train_matrix[:, 0]        
training_inputs = train_matrix[:, 1:]

testing_P_star = test_matrix[:, 0]
testing_inputs  = test_matrix[:, 1:]


# Define the model
model = Sequential([
    Dense(8, input_dim=8, activation='relu'),   # First hidden layer with 8 neurons 
    Dense(16, activation='relu'),     
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),           # Second hidden layer with 8 neurons
    Dense(1, activation='linear')                 # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(training_inputs, training_P_star, epochs=100, batch_size=8)

# Evaluate the model
loss = model.evaluate(training_inputs, training_P_star)
print(f'Loss: {loss}')

# ## single test
# x = np.array([[0.4, 0.1, 1, -1.5, 0, 0, 0, 0]])
# print(model.predict(x))

## predictions
P_star_pred = model.predict(testing_inputs)

## RMSE calculator
mean_square_error = np.square(np.subtract(testing_P_star,P_star_pred)).mean()  
root_mean_square_error = math.sqrt(mean_square_error)

## Compute RRMSE
rrmse = root_mean_square_error / np.mean(testing_P_star)

## Create a scatter plot
plt.scatter(testing_P_star,P_star_pred, color='crimson')

## Add title and labels
plt.title(f'RRMSE = {rrmse:.2g}')
plt.xlabel(r'Ground-truth $\bar P^*$')
plt.ylabel('Predictions')
## Make the plot square
plt.gca().set_aspect('equal', adjustable='box')

## Create a range for the unity line
x_unity = np.linspace(0, 0.33, 100)
y_unity = x_unity  # Unity line where y = x
## Plot the unity line
plt.plot(x_unity, y_unity, label='Unity Line (y = x)', color='black', linestyle='--')

## Show the plot
plt.show()

## Saving NN model
model.save("my_model.h5") 