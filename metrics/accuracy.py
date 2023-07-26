import numpy as np
def accuracy(y_test,y_pred):
    total_images=len(y_test)
    correct_predictions=np.sum(y_test==y_pred)
    accuracy=correct_predictions/total_images
    return accuracy
        

