from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files, get_image_label_pairs
from os.path import join 

def read_image(image_path: str) ->np.ndarray:
    """ reads image from the given path and returns as a numpy array
    """
    image = Image.open(image_path)
    
    img_array= np.asarray(image)
    
    return img_array
def display_grid(image_dir:str, images:list, labels:list, n_rows:int, n_cols:int, title:str, fig_size:tuple=(10,10)):
      """display grid of images with their labels"""
    #   image_paths= list_files(image_dir)
    #   no_of_images= n_rows * n_cols
    #   new_image_list= image_paths[0 : no_of_images]
      fig , ax = plt.subplots(n_rows, n_cols , figsize=fig_size) 
      fig.suptitle(title)
      index = 0
      for i in range(n_rows):
        for j in range(n_cols):
                
                # image= new_image_list[index]
                data_path= join(image_dir, images[index])
               
                image_array = read_image(data_path)
                ax[i][j].imshow(image_array)
                
                ax[i,j].axis('off')
                ax[i,j].set_title(labels[index])
                
                
                index += 1
      plt.show()
      
      
       
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
    LABEL= "csom"
    n_rows=3
    n_cols= 4
    image_path, labels= get_image_label_pairs(DATA_DIR, LABEL)
    
    display_grid(DATA_DIR, image_path , labels , n_rows, n_cols ,LABEL)
    
    new_image_list= image_path[0 : 12]
    fig , ax = plt.subplots(nrows =  1, ncols=12 , figsize= (10 , 10))   
    for i, image_path  in enumerate(new_image_list):
        
        data_path= join(DATA_DIR, image_path)
        image = read_image(data_path)
        ax[i].imshow(image)
        
        
   

    plt.show()
        
        
        
        
    
    
    
    # i=0
    # j=0
    # for image_path  in new_image_list:
        
    #     data_path= join(DATA_DIR, image_path)
    #     image = read_image(data_path)
    
    #     ax[i][j].imshow(image)
    #     print(i*ncols+j)
    #     j= j+1
    #     if j==ncols:
    #         i=i+1
    #         j=0
    #     if i == nrows:
    #         break       
    #plt.show()