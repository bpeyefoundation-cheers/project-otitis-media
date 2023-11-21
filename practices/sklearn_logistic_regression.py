# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_iris
# Import sklearn metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the iris dataset
iris = load_iris()
# print(iris)
# print(iris.DESCR)
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data)
# print(iris.target)

# Create a logistic regression object
lr = LR()
lr.fit(iris.data, iris.target)
Y_PRED = lr.predict(iris.data)

# Print the coefficients
print('Coefficients: \n', lr.coef_)
print('Intercept: \n', lr.intercept_)
print('Score: \n', lr.score(iris.data, iris.target))

# Print the accuracy score
print('Accuracy score: \n', accuracy_score(iris.target, Y_PRED))

# Print the confusion matrix
print('Confusion matrix: \n', confusion_matrix(iris.target, Y_PRED))

# Print the classification report
print('Classification report: \n', classification_report(iris.target, Y_PRED))

# Plot the confusion matrix
cm = confusion_matrix(iris.target, Y_PRED)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.show()

# Plot the data
plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Sepal length vs Sepal width')
plt.show()

# # Create a dataset class
# class Dataset:
#     def __init__(self, data, target):
#         self.data = data
#         self.target = target
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx], self.target[idx]
    
# # Create a dataset object
# dataset = Dataset(iris.data, iris.target)
# print(len(dataset))
# print(dataset[0])


