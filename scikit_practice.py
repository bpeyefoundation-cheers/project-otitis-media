import numpy as np
from sklearn.linear_model import LinearRegression as LR

import matplotlib.pyplot as plt

x_train= np.linspace(0,50 ,51, dtype=int).reshape(-1,1)
x_test= np.linspace(50 , 99, 51, dtype= int).reshape(-1,1)
#Y= np.dot(x, np.array([4])) + 3
m= 2
c= 2
y_train=m*x_train +c

#generate a gaussion noise
noise = np.random.normal(0,10,51).reshape(-1 , 1)

#add the noise to the data
y_train= y_train+noise

#plot the data
plt.scatter(x_train,y_train, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data Plot')
plt.xticks([i for i in range(0, 51,5)])
#plt.yticks([i for i in range(0, 51,5)])
plt.show()

#create linear regression object
lr = LR()
lr.fit(x_train, y_train)
Y_pred = lr.predict(x_test)
# lr.score(x_train , y_train)


#print the coefficients
print('Coefficients: \n', lr.coef_)
print('Intercept: \n', lr.intercept_)
print('Score: \n', lr.score(x_train, y_train)) #mean square error




