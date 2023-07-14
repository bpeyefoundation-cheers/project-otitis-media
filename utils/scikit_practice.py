import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR #LR is class

# X = np.linspace(0, 49 ,50)
# M = 2
# M = np.random.random()
# C = 5
# Y = M*X +C
# print(X)
# print(Y)
# print(type(X[0]))
#Create s linear regression object
lr = LR() #LR ma instance banako

X_train = np.linspace(0, 49 , 50)
X_test = np.linspace(50 , 99 ,51 , dtype =int)
Y_train =  X_train * 4 + 3
print(X_train.shape)
# print(Y_train.shape)
# print(X_test)

#Generation a gaussian noise
noise = np.random.normal(0, 15 , 50)

print(noise.shape)
#Add noise to the data
print(Y_train.shape)
Y_train = Y_train + noise

#plot the data
plt.figure()
plt.scatter(X_train, Y_train, color='blue' , marker= '+' , label= 'Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data')
plt.xticks([i for i in range(0,101, 10)])
# plt.yticks([i for i in range(0,101, 10)])
plt.show()

#Create a linear regression object
lr = LR()
lr.fit(X_train.reshape(-1, 1) , Y_train.reshape(-1, 1)) 
Y_pred = lr.predict(X_test)

#print the coefficient
print('Coefficients: \n' , lr.coef_)
print('Intercept: \n' , lr.intercept_)
print('Score: \n' , lr.score(X_train ,Y_train))


