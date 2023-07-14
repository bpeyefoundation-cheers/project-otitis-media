from PIL import Image
import numpy as np
from utils.io import list_files
import matplotlib.pyplot as plt
from os.path import join



def read_image(image_path : str) -> np.ndarray :
    """ Reads image from given path and returns it as numpy array"""

    image = Image.open(image_path)
    img_array = np.asarray(image)
    return img_array



if __name__ == "__main__" :
    
    # IMAGE_PATH = r'C:\Users\Dell\Desktop\Otitis Media\project-otitis-media\data\middle-ear-dataset\csom\o2.jpg'

    # image = read_image(IMAGE_PATH)

    # #remove xticks and yticks
    # plt.xticks([])
    # plt.yticks([])

    # #set ttile
    # plt.title("csom")

    # plt.imshow(image)
    # plt.show()


    
    fig , ax = plt.subplots(nrows = 3 , ncols = 4, figsize = (10,10))

    DATA_DIR = "data\middle-ear-dataset\csom"
    image_path = list_files(DATA_DIR, 'jpg') 
    # print(image_path)
    image_list = image_path[0 : 12]
    # print(image_list)

    for image_file in image_list:
        data_path = join(DATA_DIR , image_file)
        image = read_image(data_path)
        ax[image_file].imshow(image)
        plt.show()





    
