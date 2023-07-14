import os
import csv
from sklearn.model_selection import train_test_split


def list_files(dir:str , file_extension:str) ->list:
    
    if not os.path.isdir(dir):
        return None

    files = os.listdir(dir)
    return files

def get_image_label_pairs(dir:str ,label:str) ->  tuple :
    filenames = list_files(dir , ' ')

    labels = [label]*len(filenames)
    return filenames, labels

def save_as_csv(image_paths , labels, outfile):
    """SAve image path and save as csv"""
    with open(outfile , 'w' , newline= '') as f :
        writer = csv.writer(f) #obj
        writer.writerow(['file' , 'label'])
        for image_path , label in zip(image_paths , labels):
            writer.writerow([image_path, label])




        
if __name__ == '__main__':
        
    x, y = [] , []
    folder_array = ['aom', 'csom' , 'myringosclerosis' , 'Normal']
    for i in folder_array :
        image_path , label = get_image_label_pairs(f'data\middle-ear-dataset\{i}', f'{i}')
        x.extend(image_path)

        y.extend(label)

    x_train, x_test, y_train , y_test = train_test_split(x, y, train_size=0.8 , test_size=0.2) #each label ko 20 or 80% lai lincha
    save_as_csv(x_train, y_train, 'data\middle-ear-dataset/train.csv')
    save_as_csv(x_test, y_test, 'data\middle-ear-dataset/test.csv')


        
    
    images , labels = get_image_label_pairs('data/middle-ear-dataset/aom' , 'aom')
    images_L , label_L = get_image_label_pairs('data/middle-ear-dataset/csom' ,'csom')

    save_as_csv(images, labels, 'data/aom.csv')
    save_as_csv(images_L , label_L , 'data/csom.csv')

        


        