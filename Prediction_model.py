import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# List of all csv filenames
TESLA_TRAIN_DATA = 'CSV_Files/TESLA 5 years.csv'
TESLA_TEST_DATA = 'CSV_Files/TESLA last month.csv'
GM_TRAIN_DATA = 'CSV_Files/GM 5 years.csv'
GM_TEST_DATA = 'CSV_Files/GM last month.csv'
TOYOTA_TRAIN_DATA = 'CSV_Files/Toyota 5 years.csv'
TOYOTA_TEST_DATA = 'CSV_Files/Toyota last month.csv'


# Data sets for stock we are currently assessing
current_train_data = TESLA_TRAIN_DATA
current_test_data = TESLA_TEST_DATA

# Number of data points to retrieve from csv files (varies with each stock we assess)
NUM_TRAIN_DATA_POINTS = 1240
NUM_TEST_DATA_POINTS = 21

LEARNING_RATE = 0.1
NUM_EPOCHS = 100


# Function to load data that we want from csv files and return final and opening prices and volume for each day
def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name,
                       skiprows=0,
                       nrows=num_data_points,
                       usecols=['Price', 'Open', 'Vol.'])
    # Prices of stock at the end of each day
    final_prices = data['Price'].astype(str).str.replace(',','').astype(np.float)
    # Prices of stock at the beginning of each day
    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    # Volume of stock exchanged throughout the day
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_prices, opening_prices, volumes


# Function to calculate differences between opening price of the next day and final price of the current day
def calculate_price_differences(final_prices, opening_prices):
    price_differences = []
    for d_i in range(len(final_prices) - 1):
        price_difference = opening_prices[d_i + 1] - final_prices[d_i]
        price_differences.append(price_difference)
    return price_differences


def calculate_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values[a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values[a_i]:
            num_correct += 1
    return (num_correct / len(actual_values)) * 100


# Training data sets
train_final_prices, train_opening_prices, train_volumes = load_stock_data(current_train_data, NUM_TRAIN_DATA_POINTS)
train_price_differences = calculate_price_differences(train_final_prices, train_opening_prices)
train_volumes = train_volumes[:-1]

# Testing data sets
test_final_prices, test_opening_prices, test_volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
test_price_differences = calculate_price_differences(test_final_prices, test_opening_prices)
test_volumes = test_volumes[:-1]

# Building computational graph after y = Wx + b

# Used to input volumes
x = tf.placeholder(tf.float32, name='x')
# Variables that our model will change to get actual output as close to expected output as possible
W = tf.Variable([.1], name='W')
b = tf.Variable([.1], name='b')
# How our model outputs the actual values
y = W * x + b
# Used to input expected values for training purposes (shows the model what a "good" outcome is)
y_predicted = tf.placeholder(tf.float32, name='y_predicted')

# Loss function based on the difference between actual and expected outputs
loss = tf.reduce_sum(tf.square(y - y_predicted))
# Optimizer aimed at minimizing loss by changing W and b
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Session is used to actually run the nodes
session = tf.Session()
# Need to initialize global variables
session.run(tf.global_variables_initializer())
for _ in range(NUM_EPOCHS):
    # Run the optimizer which will allow it to change the values of W and b to minimize loss
    session.run(optimizer, feed_dict={x: train_volumes, y_predicted: train_price_differences})

results = session.run(y, feed_dict={x: test_volumes})
print(results)

accuracy = calculate_accuracy(test_price_differences, results)
print("Accuracy of model: {0:.2f}%".format(accuracy))

# # Plotting purposes only, not necessary
plt.figure(1)
plt.plot(train_volumes, train_price_differences, 'bo')
plt.title('Price Differences for Given Volumes for the Past Year')
plt.xlabel('Volumes')
plt.ylabel('Price differences')
plt.show()
