import numpy as np


# index =0
def accuracy(y_true: list , y_pred:list):
    
    total_labels = len(y_pred)
    correct_labels =np.sum( y_true== y_pred)
  
    accuracy= correct_labels/ total_labels
    
    return accuracy
    
#     count += [i for i in y_true[index] if i y_true[index]== y_pred[index]]
     
