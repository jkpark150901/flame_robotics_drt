
import sys, os, platform
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class AppWindow:

    def __init__(self, config:dict):
        self.__config = config
        
        self.window = gui.Application.instance.create_window(config["app_window_title"], config["screen_width"], config["screen_height"])
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        
        self.window.set_on_layout(self._on_layout)
        
        