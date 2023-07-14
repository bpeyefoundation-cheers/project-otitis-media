from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files
from os.path import join 

def read_image(image_path: str) ->np.ndarray:
    """ reads image from the given path and returns as a numpy array
    """
    image = Image.open(image_path)
    
    img_array= np.asarray(image)
    
    return img_array
   
    
if __name__ == "__main__":

    #print("lets do the pre-processing")
    # IMAGE_PATH = r'C:\Users\Dell\Desktop\Otitis_Media\project-otitis-media\data\middle-ear-dataset\csom\o1.jpg'
    # image = read_image(IMAGE_PATH)
     
    
    # #print(type(img))
    # plt.imshow(image)
    
    # #remove xticks and yticks
    # plt.xticks([])
    # plt.yticks([])
    
    # #set title
    # plt.title("csom")
    
    # plt.show()
    DATA_DIR = "data\middle-ear-dataset\csom"
    
    image_paths = list_files(DATA_DIR, '')
    
    new_image_list= image_paths[0 : 12]
    fig , ax = plt.subplots(nrows =  1, ncols=12 , figsize= (10 , 10))   
    for i, image_path  in enumerate(new_image_list):
        
        data_path= join(DATA_DIR, image_path)
        image = read_image(data_path)
        ax[i].imshow(image)
        
        
   

    plt.show()
        
        
        
        
    
    
    
    