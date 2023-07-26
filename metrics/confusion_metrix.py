import numpy as np


def confusionmatrix(y_true: list, y_pred: list):
    c_matrix = np.zeros((4,4))
    
    for i in range(4):
        for j in range(4):
            selected= (y_true == i) & (y_pred ==j )
            count = np.sum(selected)
           
            c_matrix[i][j]=  count
            
    return c_matrix
   
 
