from sklearn.linear_model import LinearRegression as LR
import numpy as np
import matplotlib.pyplot as plt

X_train = np.linspace(0, 49, 50).reshape(-1, 1)
X_test = np.linspace(50, 99, 50).reshape(-1, 1)

print(X_train)
m = 10
# m = np.random.random()
print(m)
c = 4
Y_train = m * X_train + c
# print(Y)

# generating noise
noise = np.random.normal(5, 5, 50).reshape(-1, 1)

# add the noise to data
Y_train = Y_train + noise
print(X_train)
print(X_test)
print(Y_train)

# visualize
plt.scatter(X_train, Y_train, color="red", marker="+", label="Data")
plt.xlabel("X")
plt.ylabel("Y")

# use list comprehensive too
plt.xticks([i for i in range(0, 101, 10)])
plt.title("Data")
plt.show()


# creating linear regression object
lr = LR()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
plt.scatter(X_train, Y_train, color="blue", marker="o", label="Data")
plt.plot(X_test, Y_pred, color="red", label="Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# print the coefficient
print("Coefficients: \n", lr.coef_)
print("Intercept: \n", lr.intercept_)
print("Score: \n", lr.score(X_train, Y_train))


# fit the linear regression to the data
#
# lr.fit(X,y)
