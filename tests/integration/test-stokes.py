

from utils import  set_test_dir, test_file, test_dir

import unittest
import xmlrunner    

import math


# Add content/stokes to path.
import os
import sys
stokes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","content", "stokes"))
sys.path.insert(0, stokes_dir)
print(sys.path)

import ug4py.pyugcore as ug4


class TestAll(unittest.TestCase):
    def setUp(self):
        set_test_dir("content/stokes")
       # os.environ['PYVISTA_OFF_SCREEN'] = "True"

    def tearDown(self):
        set_test_dir("../..")

    def test_nigon(self):
        from nigon import test_nigon
        
        # Low res
        errors = test_nigon(numRefs=2, lsolver=ug4.LUCPU1())
        norm = math.sqrt(errors[0]*errors[0] + errors[1]*errors[1])
        self.assertLessEqual(norm, 1e-3, "Error bounded.")

        # Medium res
        errors = test_nigon(numRefs=3, lsolver=ug4.LUCPU1())
        norm = math.sqrt(errors[0]*errors[0] + errors[1]*errors[1])
        self.assertLessEqual(norm, 1e-4, "Error bounded.")
    
   
    
        
if __name__ == "__main__":
    # unittest.main()
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports'),
        # these make sure that some options that are not applicable
        # remain hidden from the help menu.
        failfast=False, buffer=False, catchbreak=False)
