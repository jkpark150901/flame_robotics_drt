VENV_DIR := $(CURDIR)/venv
PYTHON := $(VENV_DIR)/bin/python3

# 3D Viewer with Vedo 3D Library
viewervedo:
	$(PYTHON) ./python/viewervedo.py --config $(CURDIR)/python/viewervedo.cfg

viewermujoco:
	$(PYTHON) ./python/viewermujoco.py --config $(CURDIR)/python/viewermujoco.cfg

mujoco-scene:
	$(PYTHON) ./tools/generate_mujoco_scene.py --config $(CURDIR)/python/viewermujoco.cfg --output $(CURDIR)/mjcf/scene.xml

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
