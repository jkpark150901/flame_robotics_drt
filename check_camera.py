import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
import open3d as o3d

def check_camera_api():
    app = gui.Application.instance
    app.initialize()
    win = app.create_window("Check API", 100, 100)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(win.renderer)
    
    cam = widget.scene.camera
    print(f"Camera type: {type(cam)}")
    print(f"Has set_projection: {hasattr(cam, 'set_projection')}")
    print(f"Projection Enum: {rendering.Camera.Projection.__members__}")
    
    # Try to set projection
    # cam.set_projection(rendering.Camera.Projection.Ortho, -100, 100, -100, 100, 0.1, 1000)
    
    app.quit()

if __name__ == "__main__":
    check_camera_api()
