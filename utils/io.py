import os
import csv
from sklearn.model_selection import train_test_split


def list_files(dir: str, file_extension: str) -> list:
    """give a directory,list all files with given extension"""
    if not os.path.isdir(dir):
        return None

    files = os.listdir(dir)
    return files


def get_image_label_pairs(image_dir: str, label: str) -> tuple:
    """assuming the image dir contains a single label image, create two lists.
    the first list contains filenames, the second list contains label for corresponding image
    e.g. (file1,file2,...n) , (erythroplakia, …, erythroplakia)
    """

    filenames = list_files(
        image_dir, ""
    )  # Update the file extension as per  requirement
    labels = [label] * len(filenames)
    return filenames, labels

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


def save_as_csv(image_paths, labels, outfile):
    """Assume image_paths = [file1, file2, ...filen] and labels = [label1,label2...labelk]
    Save a CSV file with a default name 'output.csv' such that each row contains:
    file1, label1
    file2, label2
    """

    # outfile = 'output.csv'

    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "label"])

        for image_path, label in zip(image_paths, labels):
            writer.writerow([image_path, label])

def save_prediction_as_csv(test_files, y_tests,y_preds, outfile):
    """Assume image_paths = [file1, file2, ...filen] and labels = [label1,label2...labelk]
    Save a CSV file with a default name 'output.csv' such that each row contains:
    file1, label1
    file2, label2
    """

    # outfile = 'output.csv'

    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["test_file", "y_test","y_pred"])

        for test_file,y_test,y_pred  in zip(test_files, y_tests,y_preds):
            writer.writerow([test_file,y_test,y_pred])



if __name__=='__main__':
        images_aom,labels_aom=get_image_label_pairs('data\middle-ear-dataset/aom','aom')
        images_csom,labels_csom=get_image_label_pairs('data\middle-ear-dataset/csom','csom')
        images_myringosclerosis,labels_myringosclerosis=get_image_label_pairs('data\middle-ear-dataset/myringosclerosis','myringosclerosis')
        images_Normal,labels_Normal=get_image_label_pairs('data\middle-ear-dataset/Normal','Normal')
        

        save_as_csv(images_aom,labels_aom,'data/aom.csv')
        save_as_csv(images_csom,labels_csom,'data/csom.csv')
        save_as_csv(images_myringosclerosis,labels_myringosclerosis,'data/myringosclerosis.csv')
        save_as_csv(images_Normal,labels_Normal,'data/Normal.csv')

        x = []
        y = []
        folders = ["aom", "csom", "myringosclerosis", "Normal"]
        for i in folders:
            images_path, label = get_image_label_pairs(f"data\middle-ear-dataset/{i}", f"{i}")
            x.extend(images_path)
            y.extend(label)
        save_as_csv(x, y, "data\middle-ear-dataset/test.csv")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        save_as_csv(x_train, y_train, "data/train.csv")
        save_as_csv(x_test, y_test, "data/test.csv")
