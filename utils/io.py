import os
import csv

def list_files(dir:str,file_extension:str) -> list:
    """ give a directory,list all files with given extension"""
    if not os.path.isdir(dir):
        return None
    
    files=os.listdir(dir)
    return files




def get_image_label_pairs(image_dir:str, label:str)  -> tuple:
    
    """ assuming the image dir contains a single label image, create two lists. 
	the first list contains filenames, the second list contains label for corresponding image 
	e.g. (file1,file2,...n) , (erythroplakia, …, erythroplakia) 
    """

    filenames = list_files(image_dir, '')  # Update the file extension as per  requirement
    labels = [label] * len(filenames)
    return filenames, labels

def save_as_csv(image_paths, labels, outfile):
    
    """ assume image_paths = [file1, file2, ...filen] and labels = [label1,label2...labelk] 
	save a csv file with filename outfile such that each row contains 
	file1, label1 
	file2, label2 
	"""

    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'Label'])  # Write the header row

        for image_path, label in zip(image_paths, labels):
            writer.writerow([image_path, label])

    