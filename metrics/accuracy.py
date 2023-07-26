import numpy as np
def accuracy(y_test , y_pred):
    
    
    total_accurate_prediction = np.sum(y_test == y_pred)
    calculated_accuracy = total_accurate_prediction / len(y_test)
    
    return(calculated_accuracy)

 
