import unittest
from utils.io import list_files , get_image_label_pairs


class TestIO(unittest.TestCase):
    def test_list_files(self):

        self.assertIsNotNone(list_files('data\middle-ear-dataset/aom' , '.tiff')) #none hunuparne tara tya datset cha
        self.assertIsNone(list_files('data\middle-ear-dataset/ao' , '.tiff'))
        # self.assertIsNotNone(list_files('data\middle-ear-dataset/ao' , '.tiff'))
        self.assertEqual(len(list_files('data\middle-ear-dataset/aom' , '')), 119)  #DATASET 119 CHA SO TEST CASE IS PASSED
        
    def test_get_image_label_pairs(self):
        self.assertIsNotNone(get_image_label_pairs('data\middle-ear-dataset/aom' , 'aom'))
        self.assertEqual(len(get_image_label_pairs('data\middle-ear-dataset/aom' , 'aom')[0]), 119)


    
        
        

