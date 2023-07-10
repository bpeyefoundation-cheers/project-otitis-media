import unittest
from my_function import add
class Test_my_functions(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(5,6),11)
        self.assertEqual(add(5,0),5)
