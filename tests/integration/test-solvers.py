import os
import sys

import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_solvers.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    #
    os.environ['PYVISTA_OFF_SCREEN'] = "True"

    # Change to the specified directory
    try:
        os.chdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
    except NotADirectoryError:
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied to access '{directory}'.")
        sys.exit(1)

    # List files and perform a job on each
    try:
        for filename in os.listdir():
            if os.path.isfile(filename):  # Check if it's a file
                # Replace the next line with the job you want to perform
                if (filename.endswith(".py")):
                    print(f"Processing file: {filename}")
                    process = subprocess.run("ipython "+ filename, shell=True, capture_output=True, text=True)
                    if process.returncode != 0:
                        print(process.returncode)
                        sys.exit(1)

                else:
                    print(f"Skipping file: {filename}")                
		# Example job: print the file's name
                # You can add more complex processing here
    except Exception as e:
        print(f"Error while processing files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

