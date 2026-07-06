# Set this to a specific interpreter when using conda or a custom venv.
# Example: PYTHON_CONFIG := C:/Users/admin/miniforge3/envs/drt/python.exe
PYTHON_CONFIG :=

VENV_DIR := $(CURDIR)/venv
ifneq ($(strip $(PYTHON_CONFIG)),)
PYTHON := $(PYTHON_CONFIG)
else
ifeq ($(OS),Windows_NT)
PYTHON := $(VENV_DIR)/Scripts/python.exe
else
PYTHON := $(VENV_DIR)/bin/python
endif
endif
PYTHON_CMD := "$(PYTHON)"

# 3D Viewer with Vedo 3D Library
viewervedo:
	$(PYTHON_CMD) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg

# !! Deprecated
controller:
	$(PYTHON_CMD) ./python/controller.py --config $(CURDIR)/python/controller.cfg

# External interface proxy
zproxy:
	$(PYTHON_CMD) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg

# Simulation/Real Control Box
simtool:
	$(PYTHON_CMD) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg

# Cobot Calibration & Verification Tool (robot SDK + NatNet)
verifycobot:
	$(PYTHON_CMD) ./python/verifycobot.py --config $(CURDIR)/python/verifycobot.cfg

# Positioner Trajectory Verify (NatNet만 필요, 로봇 SDK 불필요)
verifypositioner:
	$(PYTHON_CMD) ./python/verifypositioner.py --config $(CURDIR)/python/verifypositioner.cfg

# Run in parallel
run:
	$(PYTHON_CMD) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg &
	$(PYTHON_CMD) ./python/viewero3d.py --config $(CURDIR)/python/viewervedo.cfg &
	$(PYTHON_CMD) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg 
