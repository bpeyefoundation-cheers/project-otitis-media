import numpy as np

def confusion_matrics(y_test , y_pred) :
    # y_test_label = list(set(y_test))
    # y_pred_label = list(set(y_pred))
    matrix = np.zeros(4, 4)
    for act_label in range(4):
        for pred_label in range (4):
            label = y_test[act_label] 
            

    

    





