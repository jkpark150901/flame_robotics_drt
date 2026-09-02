# Set this to a specific interpreter when using conda or a custom venv.
# Example: PYTHON_CONFIG := C:/Users/admin/miniforge3/envs/drt/python.exe


PYTHON := /home/user/miniforge3/envs/drt/bin/python

PYTHON_CMD := "$(PYTHON)"

# 3D Viewer with Vedo 3D Library
viewervedo:
	$(PYTHON_CMD) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg

# Standalone pose/path planning service
robotcore:
	$(PYTHON_CMD) ./python/robotcore.py --config $(CURDIR)/python/viewervedo.cfg

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

# Run in parallel: zproxy + standalone robotcore + viewer(external) + simtool
run:
	$(PYTHON_CMD) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg &
	$(PYTHON_CMD) ./python/robotcore.py --config $(CURDIR)/python/viewervedo.cfg &
	$(PYTHON_CMD) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg --robot_core_mode external &
	$(PYTHON_CMD) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg

# Run in parallel: zproxy + viewer with embedded child robotcore + simtool
run-embedded:
	$(PYTHON_CMD) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg &
	$(PYTHON_CMD) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg --robot_core_mode embedded &
	$(PYTHON_CMD) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg
