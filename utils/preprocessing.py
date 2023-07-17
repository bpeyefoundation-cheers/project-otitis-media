from PIL import Image , ImageOps
import numpy as np
from utils.io import list_files , read_as_csv
import matplotlib.pyplot as plt
from os.path import join





def read_image(image_path : str , mode:str ,size = (256, 256) ) -> np.ndarray :
    """ Reads image from given path and returns it as numpy array
    
    TODO: resize to a default value (256,256)
    
    image_path : str file_path of the image which we want to read
    mode : str either one of the 'zoom' or 'pad'. this argument determines if the image is either zoomed in or padded at the border
      """

    image = Image.open(image_path) 
    print(image.size)
    height = image.height
    width = image.width
    left = 0
    top = 0
    right = image.width
    bottom = image.height
    print(height, width)
    if image.height == image.width :
         mode = None
    elif mode == 'zoom':
        if height > width :
             size_diff = height - width
             top = top - (size_diff/2)
             bottom = bottom - (size_diff/2)
            #  print(image.size)
        else :
             size_diff = width - height
             left = left - (size_diff/2)
             right = right - (size_diff/2)
            #  print(image.size)
             
        zoomed_image = image.crop((left, top , right, bottom))

    elif mode == 'padding' :
        padded_image =ImageOps.pad(image, (256 ,256), color =None , centering = (0.5 , 0,5))
        # print(image.size)
    

    

             
             
    image = image.resize(size)
    img_array = np.asarray(image)
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
                data_path= join(image_dir, labels[index], images[index])
               
                image_array = read_image(data_path , 'padding')
                ax[i][j].imshow(image_array)
                
                ax[i,j].axis('off')
                ax[i,j].set_title(labels[index])
                
                
                index += 1
      plt.show()



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

     # DATA_DIR = "data/middle-ear-dataset/aom"
    # LABEL= "aom"
    DATA_DIR= "data/middle-ear-dataset"
    n_rows=3
    n_cols= 3
    LABEL= "middle-ear-dataset"
    image_path, labels= read_as_csv("data/test.csv")
    
    #get_image_label_pairs(DATA_DIR, LABEL)
    
    display_grid(DATA_DIR, image_path , labels , n_rows, n_cols ,LABEL)
    
    # new_image_list= image_path[0 : 12]
    # fig , ax = plt.subplots(nrows =  1, ncols=12 , figsize= (10 , 10))   
    # for i, image_path  in enumerate(new_image_list):
        
    #     data_path= join(DATA_DIR, image_path)
    #     image = read_image(data_path)
    #     ax[i].imshow(image)
        
    # plt.show()
        
        
        
        
    
    
    
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

    
   




    
