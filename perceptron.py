import numpy as np

def calculate_weighted_sum( input , weights, intercept):
    weighted_sum = np.dot(input , weights) + intercept 
    return weighted_sum

def perceptron_output(input, weights, intercept):
     weight = calculate_weighted_sum(input , weights, intercept)
     value= np.where( weight < 0 , -1 , 1)
     return value

def perceptron_learning( inputs , weights , intercept , target):
    output = perceptron_output(inputs , weights , intercept)
    error = np.where( target != output ,  1 , 0 ) * target
    
    #n_weight  = weights * error
    #new_weight = np.dot(inputs , n_weight)
    new_weight = np.dot( inputs , error ) + weights 
    updated_intercept = intercept + error
    return new_weight , updated_intercept
    
    
     

if __name__ == "__main__":
    input=np.array([[1,2],[2,2],[4,5],[6,6]])
    target= np.array([1, 1, -1, -1])
    epoch = 100
     
    weights = np.array([0, 0])
    '''bias and intercept are same '''

    intercept= 0  
    
    for loop in range(epoch):
        for inp, tar in zip(input , target):
            weights, intercept= perceptron_learning(inp, weights, intercept, tar)
            
     #print(final_weight , final_intercept)
    inp= input[0]
    output = perceptron_output(inp , weights, intercept)
    print(output)