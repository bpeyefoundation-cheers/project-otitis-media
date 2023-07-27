
import numpy as np
#from evaluate import y_test,y_pred

def confusion_metrics_calculate(y_test,y_pred):
    y_test_label=list(set(y_test))
    y_pred_label=list(set(y_pred))
    
    confusion_matrix=np.zeros((len(y_test_label),len(y_pred_label)))
    for actual_label, predicted_label in zip(y_test, y_pred):
        true_idx = y_test_label.index(actual_label)
        pred_idx = y_pred_label.index(predicted_label)
        confusion_matrix[true_idx][pred_idx] += 1

    return confusion_matrix

    # count=0
    #for predicted_label,actual_label in zip(y_test,y_pred):

    #     for actual_label in y_pred:
    #         if y_test_label[idx]==y_pred_label[idx]:
    #             count+=1
    #         else:
    #             np.sum(if y_pred[0]==y_test[1])



   

            
# if __name__=="__main__":
#     out=confusion_metrics_calculate(y_test,y_pred)
#     print("confusion",out)
    
