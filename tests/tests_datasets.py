from datasets.image_datasets import OtitisMediaDataset

dataset=OtitisMediaDataset(csv_file=r'data\train.csv')

print(f'This dataset has{len(dataset)} images/labels')
print(f'A sample from this datadet has shape {dataset[0][0].shape} and label{dataset[0][1]}')