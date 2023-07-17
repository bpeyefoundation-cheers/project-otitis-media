from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files, get_image_label_pairs
from os.path import join 

def read_image(image_path: str , size:tuple=(256,256)) ->np.ndarray:
    """ reads image from the given path and returns as a numpy array
    """

    image = Image.open(image_path)
    image= image.resize(size)
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
    
    
    LABEL ="csom"
    DATA_DIR = "data\middle-ear-dataset\csom"
    image_path , labels= get_image_label_pairs(DATA_DIR, LABEL)
    n_rows = 3
    n_cols=4
    
    display_grid(DATA_DIR, image_path , labels , n_rows, n_cols ,LABEL)
    
    # image_paths = list_files(DATA_DIR, '')
    # nrows=4
    # ncols=4
    # no_of_images= nrows*ncols
    # new_image_list= image_paths[0 : no_of_images]
    # print(new_image_list[0])
    # fig , ax = plt.subplots(nrows, ncols , figsize= (10 , 10))   
    # fig.suptitle("CSOM")
    # index = 0
    # for i in range(nrows):
    #     for j in range(ncols):
    #         #index= i*ncols+j
    #         #print(new_image_list[index])
    #         image= new_image_list[index]
    #         data_path= join(DATA_DIR, image)
    #         print(data_path)
    #         image_array = read_image(data_path)
    #         ax[i][j].imshow(image_array)
    #         # ax[i][j].set_xticks([])
    #         # ax[i][j].set_yticks([])
    #         ax[i,j].axis('off')
    #         ax[i,j].set_title(f"Label: CSOM {index}")
            
            
    #         index += 1
    
    
    
    
    
    
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