import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

training_data = pd.read_excel("train_data_lagged.xlsx")
validation_data = pd.read_excel("validation_data_lagged.xlsx")

# ask user for learning parameter and epoch count
learning_parameter = float(input("Enter the learning parameter value: "))
start_param = learning_parameter
# end_param = float(input("Enter end paramter for annealing: "))
num_epochs = int(input("Enter the number of epochs: "))
momentum_constant = 0.9

# number of neurons at each layer
input_size = 8
hidden_size = int(input("Enter the number of hidden nodes: "))
output_size = 1

# prepare data arrays
X = training_data.iloc[:, :input_size].to_numpy()  # extract all input features as input data
y = training_data.iloc[:, input_size].to_numpy()   # extract predictand column

X_val = validation_data.iloc[:, :input_size].to_numpy()
y_val = validation_data.iloc[:, input_size].to_numpy()

# set X, y min and max 
X_min, X_max = X.min(axis=0), X.max(axis=0)
y_min, y_max = y.min(), y.max()

def normalise(X, X_min, X_max):
    return 0.7*(X - X_min) / (X_max - X_min) + 0.3

def denormalize(y_norm, y_min, y_max):
    return ((y_norm-0.3)/0.7) * (y_max - y_min) + y_min

# normalize training data (min-max scaling)
X = normalise(X, X_min, X_max)
y = normalise(y, y_min, y_max)

# normalize validation data (min-max scaling)
X_val = normalise(X_val, X_min, X_max)
y_val = normalise(y_val, y_min, y_max)

# initialize weights and biases to random variables using formula (-2/n, 2/n) as the range of random values
W1 = np.random.uniform(-2/input_size, 2/input_size, (hidden_size, input_size))  
b1 = np.random.uniform(-2/input_size, 2/input_size, (hidden_size, 1))
W2 = np.random.uniform(-2/hidden_size, 2/hidden_size, (output_size, hidden_size))  
b2 = np.random.uniform(-2/hidden_size, 2/hidden_size, (output_size, 1))            

def calculate_omega(W1, W2):
    n = W1.size + W2.size
    omega = (1 / (2 * n)) * (np.sum(W1**2) + np.sum(W2**2))
    return omega

# function to calculate the weighted sum
def weighted_sum(W, x, b):
    return np.dot(W, x) + b

# function to calculate sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def annealing_func(x):
    return end_param + (start_param - end_param) * (1 - (1 / (1 + np.exp(10 - ((20 * x) / num_epochs)))))

def predict(X_new, W1, b1, W2, b2):
    predictions = []
    for row in X_new:
        x = row.reshape(-1, 1)
        Z1 = np.dot(W1, x) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        predictions.append(A2.item())
    return np.array(predictions)

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def save_best_model(W1, b1, W2, b2):
    return {
        'W1': W1.copy(),
        'b1': b1.copy(),
        'W2': W2.copy(),
        'b2': b2.copy(),
    }

train_errors = []
validation_errors = []
best_epoch = 0
best_validation_error = float('inf')
best_weights = None

progress_bar = tqdm(range(num_epochs), desc="Training", leave=True)

# training loop over set epochs user inputs
for epoch in progress_bar:
    epoch_train_losses = []

    # loop over each training sample (gradient descent)
    for i in range(len(X)):
        x = X[i].reshape(input_size, 1)
        predictand = y[i].reshape(1, 1)

        # forward pass
        Z1 = weighted_sum(W1, x, b1)  # hidden layer weighted sum
        A1 = sigmoid(Z1)              # hidden layer activation

        Z2 = weighted_sum(W2, A1, b2) # output layer weighted sum
        A2 = sigmoid(Z2)              # final prediction

        # backward pass
        delta_output = (predictand - A2) * (A2 * (1 - A2))
        delta_hidden = np.dot(W2.T, delta_output) * (A1 * (1 - A1))

        # update weights
        W2_new = W2 + learning_parameter * np.dot(delta_output, A1.T)
        b2 = b2 + learning_parameter * delta_output

        W1_new = W1 + learning_parameter * np.dot(delta_hidden, x.T)
        b1 = b1 + learning_parameter * delta_hidden

        # momentum
        W2_change = W2_new - W2
        W1_change = W1_new - W1

        W2 = W2_new + W2_change * momentum_constant
        W1 = W1_new + W1_change * momentum_constant

    current_train_error = np.mean(epoch_train_losses)

    # Adaptive learning rate adjustment (fixed bug: make sure train_error is available)
    if epoch >= 250 and epoch % 250 == 0:
        prev_train_error = train_errors[epoch - 250]
        if current_train_error < prev_train_error:
            learning_parameter *= 1.05
        else:
            learning_parameter *= 0.7
        learning_parameter = max(0.01, min(learning_parameter, 0.5))

    # Calculate and store training error after each epoch
    train_predictions = predict(X, W1, b1, W2, b2)
    train_error = calculate_mse(y, train_predictions)
    train_errors.append(train_error)

    # Calculate and store validation error after each epoch
    validation_predictions = predict(X_val, W1, b1, W2, b2)
    validation_error = calculate_mse(y_val, validation_predictions)
    validation_errors.append(validation_error)

    # Check if this is the best model so far (based on validation error)
    if validation_error < best_validation_error:
        best_validation_error = validation_error
        best_epoch = epoch
        best_weights = save_best_model(W1, b1, W2, b2)

    progress_bar.set_postfix({
        "Train MSE": f"{train_error:.6f}",
        "Val MSE": f"{validation_error:.6f}",
        "Best Val": f"{best_validation_error:.6f}"
    })

progress_bar.close()


plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Training Error', color='blue')
plt.plot(validation_errors, label='Validation Error', color='red')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Validation Error Over Time')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
plt.show()

test_data = pd.read_excel("test_data_lagged.xlsx")
test_data_predictors = test_data.iloc[:, :input_size].to_numpy()
test_data_predictand = test_data.iloc[:, input_size].to_numpy()

# normalise test data (min-max scaling)
test_data_predictors = normalise(test_data_predictors, X_min, X_max)

normalized_predictions = predict(test_data_predictors, best_weights["W1"], best_weights["b1"], best_weights["W2"], best_weights["b2"])

# denormalize predictions
normalized_predictions = normalized_predictions.flatten()
denormalized_predictions = denormalize(normalized_predictions, y_min, y_max)

def calculate_msre(y_true, y_pred):
    # prevent divide by zero
    epsilon = 1e-8
    relative_errors = ((y_pred - y_true) / (y_true + epsilon)) ** 2
    msre = np.mean(relative_errors)
    return msre

def calculate_ce(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if denominator == 0:
        raise ValueError("Denominator in CE calculation is zero; check y_true for constant values.")
    
    ce = 1 - (numerator / denominator)
    return ce

def calculate_rsqr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    denominator1 = np.sum((y_true - np.mean(y_true)) ** 2)
    denominator2 = np.sum((y_pred - np.mean(y_pred)) ** 2)
    
    if denominator1 == 0 or denominator2 == 0:
        raise ValueError("Denominator in RSqr calculation is zero; check y_true or y_pred for constant values.")
    
    rsqr = (numerator ** 2) / (denominator1 * denominator2)
    return rsqr

def calculate_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

test_msre = calculate_msre(test_data_predictand, denormalized_predictions)
ce_value = calculate_ce(test_data_predictand, denormalized_predictions)
rsqr_value = calculate_rsqr(test_data_predictand, denormalized_predictions)
rmse_value = calculate_rmse(test_data_predictand, denormalized_predictions)

print(f"Test MSRE: {test_msre:.6f}")
print(f"Test CE: {ce_value:.6f}")
print(f"Test R²: {rsqr_value:.6f}")
print(f"Test RMSE: {rmse_value:.6f}")

print(f"Test min: {np.min(test_data_predictand)}, Test mean: {np.mean(test_data_predictand)}, Test max: {np.max(test_data_predictand)}")
print(f"Test min: {np.min(denormalized_predictions)}, Test mean: {np.mean(denormalized_predictions)}, Test max: {np.max(denormalized_predictions)}")


def plot_graph(prediction, actual):
    plt.figure(figsize=(12, 6))
    plt.plot(prediction, label='Predicted', color='blue')
    plt.plot(actual, label='Actual', color='red')
    plt.xlabel('Time (days)')
    plt.ylabel('Flow Rate at Skelton - Cumecs (m³/s)')
    plt.title('Actual vs Predicted Flow Rates Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_graph(denormalized_predictions, test_data_predictand)