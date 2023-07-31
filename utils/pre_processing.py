from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
#from os.path import join 
#from viz.visualization import display_grid
import os


label_map={
  "aom":1,
  "csom":2,
  "myringosclerosis":3,
  "Normal":4
}

data_root = "data\middle-ear-dataset"
index_to_label_dict= { index:label for label,index in label_map.items()}

def image_transforms(file_name, label) -> np.ndarray:
    file_path = os.path.join(data_root, label, file_name)
    array = read_image(file_path, "zoom", grayscale=True)
    flatten_image = array.flatten()
    return flatten_image


def label_transforms(label) -> int:
    # label_to_index
    return label_to_index(label)
  
def label_to_index( label:str):
  if label not in label_map:
    raise KeyError("label in not valid")
  return label_map[label]

def index_to_label( inx:int):
  if inx not in index_to_label_dict:
    raise KeyError("index is not valid")
  return index_to_label_dict[inx]
  
#  (list(my_dict.keys())
#       [list(my_dict.values()).index(100)])

 
  

def read_image(image_path: str,mode:str,size:tuple=(256,256), grayscale:bool = False) ->np.ndarray:
    """ reads image from the given path and returns as a numpy array
    TODO: resize the image and implement the mode of zoom or paddding 
    args:
    -----
    image_path: the image which we want to read
    mode: either 'zoom' or 'pad'
    size:the size of the image we want to set to
    """

    image = Image.open(image_path)
    #image= image.resize(size)
    height, width= image.size
  
    if mode == "padding":
       if height== width:
         pass
       else:
        image=ImageOps.pad(image, (256, 256), color=None, centering=(0.5, 0.5))
    
    if mode== "zoom":
        diff= height-width
        if diff>0:
            right = width
            (left, upper, right, lower) = (0, diff//2, right, height-(diff//2))
            image= image.crop((left, upper, right, lower))
           
             
        else:
            lower= height
            diff = abs(diff)
            (left, upper, right, lower) = (diff//2, 0, width-(diff//2), lower)
            image = image.crop((left, upper, right, lower))
     
    image = image.resize((256,256))     
    img_array= np.asarray(image)
    return img_array

def image_transforms(file_name, label) -> np.ndarray:
    file_path = os.path.join(data_root, label, file_name)
    array = read_image(file_path, "zoom", grayscale=True)
    flatten_image = array.flatten()
    return flatten_image   
  
      
       
if __name__ == "__main__":
  index = label_to_index("aom")
  print(index)
  
  value= index_to_label(3)
  print(value)

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



    # DATA_DIR = "data/middle-ear-dataset/aom"
    # LABEL= "aom"
    # DATA_DIR= "data/middle-ear-dataset"
    # n_rows=3
    # n_cols= 3
    # LABEL= "middle-ear-dataset"
    # image_path, labels= read_as_csv("data/test.csv")
    
    # #get_image_label_pairs(DATA_DIR, LABEL)
    
    # display_grid(DATA_DIR, image_path , labels , n_rows, n_cols ,LABEL)
    
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