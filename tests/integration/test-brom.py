
from utils import  set_test_dir, test_file, test_dir

import os
import unittest
import xmlrunner    


class Bromide3D(unittest.TestCase):
    def setUp(self):
        os.environ['PYVISTA_OFF_SCREEN'] = "True"

    def tearDown(self):
        set_test_dir("../..")

    def test_bromide(self):
        set_test_dir("content/brom")
        self.assertEqual(test_file("BromDiffusion.py"), 0, "Must return true")
        
if __name__ == "__main__":
    # unittest.main()
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports'),
        # these make sure that some options that are not applicable
        # remain hidden from the help menu.
        failfast=False, buffer=False, catchbreak=False)
