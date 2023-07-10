import unittest
from utils.io import list_files

class TestIO(unittest.TestCase):
    def test_list_files(self):

        self.assertIsNotNone(list_files('data\middle-ear-dataset/aom', ''))
        self.assertEqual(len(list_files('data\middle-ear-dataset/aom', '')), 119)
        
       # self.assertIsNotNone(list_files('data\middle-ear-dataset/am', ''))
        
