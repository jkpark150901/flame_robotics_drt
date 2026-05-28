VENV_DIR := $(CURDIR)/venv
PYTHON := /home/dev/miniconda3/envs/drt/bin/python

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

# Run in parallel
run:
	$(PYTHON) ./python/zproxy.py --config $(CURDIR)/python/zproxy.cfg &
	$(PYTHON) ./python/viewero3d.py --config $(CURDIR)/python/viewervedo.cfg &
	$(PYTHON) ./python/simtool.py --config $(CURDIR)/python/simtool.cfg 