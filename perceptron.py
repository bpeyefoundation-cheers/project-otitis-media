import numpy as np
def calculate_weighted_sum(input:np.array, weight :np.array, intercept:float):
    weighted_sum = np.dot(input , weight) + intercept
    return weighted_sum


def perceptron_output(input, weight, intercept):
    weight = calculate_weighted_sum(input, weight , intercept )
    value = np.where(weight < 0 , -1, 1)
    return value


def peceptron_learning(input, weight , intercept, target):
    output = perceptron_output(input, weight, intercept)
    error = np.where(target != output, 1, 0)* target
    new_weight = np.dot(input, error) + weight

    new_intercept = intercept + error
    return(new_weight , new_intercept)
    



if __name__ == "__main__":
    input= np.array([[1,2] , [2,2] ,[4,5] , [6,6]])
    target= np.array([1, 1, -1 , -1])  
    weight = np.array([0, 0])
    intercept = 0.0
    n_epoch = 50

    for e in range(n_epoch):
    
        for i , t in zip(input , target):

    
        # final_sum = calculate_weighted_sum(input, weight , intercept)
        # output = perceptron_output(input, weight , intercept)
        # print(final_sum)
        # print(output)
            weight , intercept = peceptron_learning(i ,weight , intercept , t)
    # print(final_weight , final_intercept)

    target_value = perceptron_output(input[0] , weight , intercept)
    print(target_value)

    