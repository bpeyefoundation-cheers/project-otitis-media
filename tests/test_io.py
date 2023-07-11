import unittest
from utils.io import list_files,get_image_label_pairs,save_as_csv

class TestIO(unittest.TestCase):
    def test_list_files(self):
        self.assertIsNotNone(list_files('data\middle-ear-dataset/aom',''))
        self.assertIsNone(list_files('data\middle-ear-dataset/ao',''))

        self.assertEqual(len(list_files('data\middle-ear-dataset/aom','')),119)

    def test_get_image_label_pairs(self):
        self.assertIsNotNone(get_image_label_pairs('data\middle-ear-dataset/aom','aom'))
        self.assertEqual(len(get_image_label_pairs('data\middle-ear-dataset/aom','aom')[0]),119)
    
    def test_save_as_csv(self):
        self.assertIsNone(save_as_csv('data\middle-ear-dataset/aom','aom'))
        
    