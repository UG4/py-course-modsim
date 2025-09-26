import os
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
            print("OUT: " +process.stdout)
            print("ERR: " + process.stderr)
            return process.returncode 
        else:
            print(f"Skipping file: {filename}")     
    return 0

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