# Use the active venv, then the project venv, then Python from PATH.
# Override explicitly with: make viewervedo PYTHON=/path/to/python
PYTHON ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard $(CURDIR)/venv/bin/python),$(CURDIR)/venv/bin/python,python3))

.PHONY: viewervedo viewer robotcore controller zproxy simtool verifycobot verifypositioner run run-embedded

PYTHON_CMD := "$(PYTHON)"

# 3D Viewer with Vedo 3D Library
viewervedo:
	$(PYTHON_CMD) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg

# Short alias for the Vedo viewer
viewer: viewervedo

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
