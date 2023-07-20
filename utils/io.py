import os
import csv
from sklearn.model_selection import train_test_split





def list_files(dir: str, file_extension: str) -> list:
    """given a dorectory, list all files with given extension"""


    if not os.path.isdir(dir):
        return None

    files = os.listdir(dir)
    return files


def get_image_label_pairs(dir: str, label: str) -> tuple:
    filenames = list_files(dir, "")
    labels = [label] * len(filenames)
    return filenames, labels


def save_as_csv(img_path: str, label: str, outfile):
    """save image path and save as csv fille"""
    with open(outfile, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        for image_path, label in zip(img_path, label):
            writer.writerow([image_path, label])

def read_as_csv(csv_file):
    image_path= []
    labels= []
    with open(csv_file , 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
           image_path.append(row[0])
           labels.append(row[1])
    return image_path, labels
            
# if __name__ == '__main__':
#     images , labels= get_image_label_pairs('data\middle-ear-dataset/aom', 'aom')
#     images_1 , labels_1 = get_image_label_pairs('data\middle-ear-dataset/csom', 'csom')

#     save_as_csv(images, labels, 'data/AOM.csv')
#     save_as_csv(images_1, labels_1, 'data/CSOM.csv')

folder = ["aom", "csom", "myringosclerosis", "Normal"]


    
# x=[]
# y=[]
# for i in folder:
#     path= f'data/middle-ear-dataset/{i}'
#     images, label=get_image_label_pairs(path, f'{i}')
#     x.extend(images)
#     y.extend(label)
    
#     save_as_csv(x, y, 'data/data.csv')  

# x_train, x_test, y_train, y_test= train_test_split(x,y, stratify= y,test_size=0.2, random_state=42)
# save_as_csv(x_train, y_train, 'data/trains.csv')
# save_as_csv(x_test, y_test, 'data/test.csv')

if __name__ == "__main__":
    x = []
    y = []
    for i in folder:
        path = f"data/middle-ear-dataset/{i}"
        images, label = get_image_label_pairs(path, f"{i}")
        x.extend(images)
        y.extend(label)

        save_as_csv(x, y, "data/data.csv")



    # file= pd.read_csv('data/data.csv')
    # print(file.head())

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
    save_as_csv(x_train, y_train, "data/train.csv")
    save_as_csv(x_test,y_test, "data/test.csv")