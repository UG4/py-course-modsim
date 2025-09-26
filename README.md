# py-course-modsim
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UG4/py-course-modsim/HEAD)

This course is a Python-based course for an introduction of UG4. 

Please start the course [here](index.ipynb).

## Dependencies:
* [UG4 on GitHub](http://github.com/UG4)
* [ug4py-base on PyPi](https://pypi.org/project/ug4py-base/)
* Ubuntu packages are in binder/apt.txt
* Python packages are in binder/requirements.txt
* Installation script in binder/postBuild

## Installation:

1) Install via pip
```
pip install ug4py-base
```

2) Clone repo:
```
git clone https://github.com/UG4/py-course-modsim.git
```

3) Run examples, e.g.,:
```
cd py-course-modsim/content/tutorial-fem
python tutorial-fem-01.py
```



## Note
This is a port of the sibling [Lua course](http://github.com/UG4/modsim-course-lua).

Installation via pip does not support parallelism via MPI. In this case, a manual build is required.

´´´
cmake -DCMAKE_CXX_COMPILER=/where/is/your/mpicxx -DCMAKE_C_COMPILER=/where/is/your/mpicc -DCMAKE_BUILD_TYPE=Release -DDIM="2;3" -DCPU=1 ..
cmake -DPARALLEL=ON -DUSE_PYBIND11=ON -DCMAKE_POLICY_DEFAULT_CMP0057=NEW ..
cmake -DLimex=ON -DConvectionDiffusion=ON -DSuperLU6=ON ..
´´´