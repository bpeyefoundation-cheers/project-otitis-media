import numpy as np


def confusionmatrix(y_true: list, y_pred: list):
    c_matrix = np.zeros((4,4))
    
    for i in range(4):
        for j in range(4):
            selected= (y_true == i) & (y_pred ==j )
            count = np.sum(selected)
           
            c_matrix[i][j]=  count
            
    return c_matrix
   
def confusion_metrics_calculate(y_test,y_pred):
    y_test_label=list(set(y_test))
    y_pred_label=list(set(y_pred))
    
    confusion_matrix=np.zeros((len(y_test_label),len(y_pred_label)))
    for actual_label, predicted_label in zip(y_test, y_pred):
        true_idx = y_test_label.index(actual_label)
        pred_idx = y_pred_label.index(predicted_label)
        confusion_matrix[true_idx][pred_idx] += 1

    return confusion_matrix
 
