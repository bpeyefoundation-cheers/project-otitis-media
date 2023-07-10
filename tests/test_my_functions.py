import unittest
from my_functions import add
class test_my_functions(unittest.Testcase):
    def test_add(self):
        self.assertEqual(add(5,6) , 11)
        self.assertEqual(add(5,0) , 5)