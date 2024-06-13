import os
import sys

import unittest
import subprocess


def set_test_dir(directory):
    # Change to the specified directory
    try:
        os.chdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return False 
    except NotADirectoryError:
        print(f"Error: '{directory}' is not a directory.")
        return False 
    except PermissionError:
        print(f"Error: Permission denied to access '{directory}'.")
        return False 
    return True


def test_file(filename):
    print(f"test_File: {filename}")
    if os.path.isfile(filename):  # Check if it's a file
        # Replace the next line with the job you want to perform
      
        if (filename.endswith(".py")):
            print(f"Processing file: {filename}")
            process = subprocess.run("ipython "+ filename, shell=True, capture_output=True, text=True)
            if process.returncode != 0:
                print(process.returncode)
                return False 
        else:
            print(f"Skipping file: {filename}")
            return True      
 
    return True

def test_dir(directory):
    # Change to the specified directory
    set_test_dir(directory)

    # List all files and perform a job on each
    try:
        for filename in os.listdir():
            if not test_file(filename):
                return False
    except Exception as e:
        print(f"Error while processing files: {e}")
        return False 
    
    return True 

# This is the 'old' main.
def main():
    if len(sys.argv) != 2:
        print("Usage: python test_solvers.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not testdir(directory):
        sys.exit(1)


class TestAll(unittest.TestCase):
    def setUp(self):
        os.environ['PYVISTA_OFF_SCREEN'] = "True"
        set_test_dir("content/tutorial-solver")

    def tearDown(self):
        set_test_dir("../..")

    #def test_solvers(self):
         # self.assertEqual(test_dir("content/tutorial-solver"), True, "Must return true")  

    def test_smoothers(self):
        self.assertTrue(test_file("example01-smoothers.py"), "Must return true")
    
    def test_multigrid(self):
        self.assertTrue(test_file("example03-multigrid.py"), "Must return true")
        
if __name__ == "__main__":
    # main()
    unittest.main()

