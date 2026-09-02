import open3d.visualization as vis
import open3d.visualization.gui as gui

print(f"vis.O3DVisualizer exists: {hasattr(vis, 'O3DVisualizer')}")
print(f"gui.O3DVisualizer exists: {hasattr(gui, 'O3DVisualizer')}")
print(f"gui.SceneWidget exists: {hasattr(gui, 'SceneWidget')}")
