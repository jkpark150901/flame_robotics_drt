# flame_robotics_drt

## Setup

## Setup Environments (Python 3.10)

1. Install python packages
```
$ pip install -r requirements.txt
```

If you use conda or a custom Python environment, set the interpreter path at the
top of `Makefile` or `run.bat`.

```
PYTHON_CONFIG := C:/Users/admin/miniforge3/envs/drt/python.exe
```

```
set "PYTHON_CONFIG=C:\Users\admin\miniforge3\envs\drt\python.exe"
```

2. Launch 3D Viewer on python virtual environment
* Unbuntu
```
(venv)$ make viewer
```
* windows
```
(venv)\ run.bat viewer
```

3. Launch Simulation Toolbox on python virtual environment
* Ubuntu
```
(venv)$ make simtool
```
* Windows
```
(venv)\ run.bat simtool
```
