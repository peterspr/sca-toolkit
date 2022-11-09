import unittest

# import python_file_name.py

class TestNotScared(unittest.TestCase):

    def test_example(self):
        self.assertTrue(2==2)

if __name__ == '__main__':
    unittest.main()