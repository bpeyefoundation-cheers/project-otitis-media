import glob
import pandas as pd
BASE_DIR="data\Datos"
TRAIN_DIR="data\Datos\Training-validation"
TEST_DIR="data\Datos\Testing"


train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")
for i in train_images:
    label=i.split("\\")[3]
    print(label)


# train_data = pd.DataFrame({'image_path': train_images,'labels':labels})

# train_data.to_csv(os.path.join(BASE_DIR, 'train.csv'), index=False)




# import csv
# def list_files(dir, file_extension):
#     if not os.path.isdir(dir):
#         return None
#     files = os.listdir(dir)
#     return files

# def get_image_label_pairs(base_dir, sub_dir, file_extension):
#     sub_dir = os.path.join(base_dir, sub_dir)
#     classes = os.listdir(sub_dir)

#     filenames = []
#     labels = []

#     for class_name in classes:
#         class_dir = os.path.join(sub_dir, class_name)

#         for file_name in os.listdir(class_dir):
#             filenames.append(os.path.join(sub_dir, class_name, file_name))
#             labels.append(class_name)

#     return filenames, labels

# base_dir = r'C:\Users\Dell\Desktop\projects\otitis-media\data\Datos'
# Training_validation = 'Training-validation'
# Testing = 'Testing'
# file_extension = '.jpg'  # Update the file extension as per your requirement

# train_filenames, train_labels = get_image_label_pairs(base_dir, Training_validation, file_extension)
# test_filenames, test_labels = get_image_label_pairs(base_dir, Testing, file_extension)

# # print("Training filenames:", train_filenames)
# # print("Training labels:", train_labels)
# # print("Testing filenames:", test_filenames)
# # print("Testing labels:", test_labels)

# def read_as_csv(csv_file):
#     image_path= []
#     labels= []
#     with open(csv_file , 'r') as f:
#         reader = csv.reader(f)
#         next(reader)
#         for row in reader:
#            image_path.append(row[0])
#            labels.append(row[1])
#     return image_path, labels
# def save_as_csv(image_paths, labels, outfile):
    

#      with open(outfile, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["file", "label"])

#         for image_path, label in zip(image_paths, labels):
#             writer.writerow([image_path, label])





# folders = ["Training-validation", "Testing"]
# x = []
# y = []

# # Iterate through folders
# for i in folders:
#     images_path, label = get_image_label_pairs("C:/Users/Dell/Desktop/projects/otitis-media/data/Datos", i, ".jpg")
#     x.extend(images_path)
#     y.extend(label)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Save the data into CSV files
# save_as_csv(x_train, y_train, "C:/Users/Dell/Desktop/projects/otitis-media/data/Datos/train.csv")
# save_as_csv(x_test, y_test, "C:/Users/Dell/Desktop/projects/otitis-media/data/Datos/test.csv")
# train_csv=r"data\Datos\train.csv"
# test_csv=r"data\Datos\test.csv"
# print(len(train_csv))