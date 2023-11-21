import numpy as np
import  matplotlib.pyplot as plt
#using class
class Perceptron:
    def __init__(self,input) :
        self.weight=np.array([0,0])
        self.intercept=0
        
    def calc_weight_sum(self,input:np.array):
    
        weighted_sum=np.dot(input,self.weight)+self.intercept
        return weighted_sum
    def predict(self,input):
        weighted_sum=self.calc_weight_sum(input)
        percep_output=np.where(weighted_sum<0,-1,1)
        return percep_output
    def fit(self,input,target,epochs):
        for _ in range(epochs):
            for x, y in zip(input, target):
                output = self.predict(x)
                error = np.where(y != output, 1, 0) * y
                self.weight += x * error
                self.intercept += y

    def plot_decision_boundary(self, inputs:np.ndarray, targets:np.ndarray) -> None:
    # Generate data points for the decision boundary line
        x_vals = np.linspace(np.min(inputs[:, 0]), np.max(inputs[:, 0]), 100)
        y_vals = -(self.weight[0] / self.weight[1]) * x_vals - (self.intercept / self.weight[1])

        # Plot the data points
        classes = np.unique(targets)
        
        assert len(classes) == 2, "Only binary classification is supported"

        plt.scatter(inputs[targets == classes[0]][:, 0], inputs[targets == classes[0]][:, 1], c='b', label=f'Class {classes[0]}')
        plt.scatter(inputs[targets == classes[1]][:, 0], inputs[targets == classes[1]][:, 1], c='r', label=f'Class {classes[1]}')

        # Plot the decision boundary line
        plt.plot(x_vals, y_vals, 'g', label='Decision Boundary')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title('Decision Boundary Plot')
        plt.grid(True)
        plt.show()
         
if __name__ == "__main__":
        input_data = np.array([[1, 2], [3, 4], [5, 6], [4, 5]])
        target = np.array([1, 1, -1, -1])

        perceptron = Perceptron(input_data)

        epochs = 10
        perceptron.fit(input_data, target, epochs)
        perceptron.plot_decision_boundary(input_data, target)

        print("Updated weight:", perceptron.weight)
        print("Updated intercept:", perceptron.intercept)