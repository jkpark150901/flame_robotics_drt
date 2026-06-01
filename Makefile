VENV_DIR := $(CURDIR)/venv
PYTHON := $(VENV_DIR)/bin/python

# 3D Viewer with Vedo 3D Library
viewervedo:
	$(PYTHON) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg

# !! Deprecated
controller:
	$(PYTHON) ./python/controller.py --config $(CURDIR)/python/controller.cfg

# External interface proxy
zproxy:
	$(PYTHON) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg

# Simulation/Real Control Box
simtool:
	$(PYTHON) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg

# Cobot Calibration & Verification Tool (robot SDK + NatNet)
verifycobot:
	$(PYTHON) ./python/verifycobot.py --config $(CURDIR)/python/verifycobot.cfg

# Positioner Trajectory Verify (NatNet만 필요, 로봇 SDK 불필요)
verifypositioner:
	$(PYTHON) ./python/verifypositioner.py --config $(CURDIR)/python/verifypositioner.cfg

# Run in parallel
run:
	$(PYTHON) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg &
	$(PYTHON) ./python/viewero3d.py --config $(CURDIR)/python/viewervedo.cfg &
	$(PYTHON) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg 