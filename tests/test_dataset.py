from datasets.image_dataset import OtitisMediaClassifier

dataset = OtitisMediaClassifier(csv_file= r'data\train.csv')

print(f'this dataset has {len(dataset)} image/labels')
print(f'A sample from this dataset has shape {dataset[0][0]} and label {dataset[0][1]}')