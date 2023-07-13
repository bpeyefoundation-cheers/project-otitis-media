import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

X_TRAIN = np.linspace(0, 49, 50).reshape(-1, 1)
X_TEST = np.linspace(50, 99, 50).reshape(-1, 1)
Y_TRAIN = X_TRAIN * 4 + 3
# print(X_TRAIN)
# print(Y_TRAIN)
# print(X_TEST)

# Generate a gaussian noise
noise = np.random.normal(5, 20, 50).reshape(-1, 1)

# Add the noise to the data 
Y_TRAIN = Y_TRAIN + noise

# Plot the data
# plt.figure()
# plt.scatter(X_TRAIN, Y_TRAIN, color='blue', marker='o', label='Data')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Data')
# plt.xticks([i for i in range(0, 101, 10)])
# # plt.yticks([i for i in range(0, 101, 10)])
# plt.show()

# Create a linear regression object
lr = LR()
lr.fit(X_TRAIN, Y_TRAIN)
Y_PRED = lr.predict(X_TEST)

# Print the coefficients
print('Coefficients: \n', lr.coef_)
print('Intercept: \n', lr.intercept_)
print('Score: \n', lr.score(X_TRAIN, Y_TRAIN))

# Get predictions
Y_TRAIN_PRED = lr.predict(X_TRAIN)
X_ALL = np.concatenate((X_TRAIN, X_TEST), axis=0)
Y_ALL = np.concatenate((Y_TRAIN_PRED, Y_PRED), axis=0)


# Plot the data
plt.figure()
plt.scatter(X_TRAIN, Y_TRAIN, color='blue', marker='o', label='Data')
plt.scatter(X_TEST, Y_PRED, color='red', marker='+', label='Linear Regression')
plt.plot(X_ALL, Y_ALL, color='green', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.xticks([i for i in range(0, 101, 10)])
# plt.yticks([i for i in range(0, 101, 10)])
plt.show()


