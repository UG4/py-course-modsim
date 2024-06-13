#!/bin/bash

# Convert all notebooks to python
find $1 -type f -name "*.ipynb" -print0 | xargs -0 -J % jupyter-nbconvert % --to python

