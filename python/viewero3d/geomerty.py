
import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
from util.logger.console import ConsoleLogger

console = ConsoleLogger.get_logger() # logger


class geometryAPI:
    def __init__(self):
        self.__console = ConsoleLogger.get_logger() # logger
        self.__geometry_container = {}
        self.__geometry_container_original = {}  # Store original geometry data for transform reset

    def destory(self):
        self.__geometry_container.clear()
        self.__geometry_container_original.clear()


    def API_add_coord_frame(self, vis, name:str, pos:list=[0,0,0], ori:list=[0,0,0], size:float=0.1) -> None:
        """ add coordinate frame """
        self.__console.debug(f"Call API_add_coord_frame : {name}")
        obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=pos)
        R_matrix = obj.get_rotation_matrix_from_axis_angle(rotation=ori) # rotation
        obj.rotate(R=R_matrix)
        obj.translate(translation=pos, relative=False)
        if not name in self.__geometry_container.keys():
            self.__geometry_container[name] = obj
            vis.add_geometry(obj, reset_bounding_box=True)
        else:
            self.__console.debug(f"({self.__class__.__name__}) Already added {name} in 3D window space")

    def API_remove_geometry(self, scene, name:str) -> None:
        """ Remove geometry """
        self.__console.debug(f"Call API_remove_geometry : {name}")
        if name in self.__geometry_container.keys():
            scene.scene.remove_geometry(name)
            del self.__geometry_container[name]
            # Also remove original data
            if name in self.__geometry_container_original:
                del self.__geometry_container_original[name]
        else:
            self.__console.debug(f"({self.__class__.__name__}) Nothing to remove the {name} in 3D window space")

    def API_clear_all_geometry(self, scene) -> None:
        self.__console.debug(f"Call API_clear_all_geometry")
        geometry_names = list(self.__geometry_container.keys())  # Create a copy of keys
        for geometry_name in geometry_names:
            self.__console.debug(f"Removing geometry: {geometry_name}")
            scene.scene.remove_geometry(geometry_name)
            del self.__geometry_container[geometry_name]
            # Also clear original data
            if geometry_name in self.__geometry_container_original:
                del self.__geometry_container_original[geometry_name]
    
    def API_add_pcd(self, scene, name:str, path:str, pos:list=[0,0,0], ori:list=[0,0,0], color:list=[0,0,0], point_size:float=1.0):
        """ add pcd geometry """
        self.__console.debug(f"Call API_add_pcd : {name} ({path})")
        obj = o3d.io.read_point_cloud(path)
        if not obj.has_points():
            self.__console.error(f"({self.__class__.__name__}) PCD has no points")
            return None
        
        # Store original data before any transforms
        original_points = np.asarray(obj.points).copy()
        
        R_matrix = obj.get_rotation_matrix_from_axis_angle(rotation=ori) # rotation
        obj.rotate(R=R_matrix)
        obj.translate(translation=pos, relative=False) # translation
        
        num_points = np.asarray(obj.points).shape[0]
        black_colors = np.tile(color, (num_points, 1))  # shape: (N, 3)
        obj.colors = o3d.utility.Vector3dVector(black_colors)

        if not name in self.__geometry_container.keys():
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = point_size  # Set point size
            scene.scene.add_geometry(name, obj, mat)
            self.__geometry_container[name] = obj
            
            # Store original data for efficient transform updates
            self.__geometry_container_original[name] = {
                'path': path,
                'original_points': original_points,
                'color': color,
                'point_size': point_size,
                'type': 'pcd'
            }
        else:
            print(f"[API_add_pcd] Already added {name} in geomery container")

    def API_update_geometry_transform(self, scene, name: str, pos: list = [0, 0, 0], ori: list = [0, 0, 0]):
        """Update geometry transform (position and orientation) efficiently"""
        self.__console.debug(f"Call API_update_geometry_transform : {name}")
        
        if name not in self.__geometry_container:
            self.__console.warning(f"Geometry {name} not found in container")
            return
        
        if name not in self.__geometry_container_original:
            self.__console.warning(f"Original data for {name} not found, using fallback method")
            self._fallback_transform_update(scene, name, pos, ori)
            return
        
        obj = self.__geometry_container[name]
        original_data = self.__geometry_container_original[name]
        
        # Create transformation matrix for absolute world coordinates
        transform_matrix = np.eye(4)
        
        # Apply rotation (absolute orientation in world space)
        R_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(ori)
        transform_matrix[:3, :3] = R_matrix
        
        # Apply translation (absolute position in world space)
        transform_matrix[:3, 3] = pos
        
        # Use set_geometry_transform for efficient update without remove/re-add
        scene.scene.set_geometry_transform(name, transform_matrix)
        
        self.__console.debug(f"Efficiently updated geometry {name} transform: pos={pos}, ori={ori}")
    
    def _fallback_transform_update(self, scene, name: str, pos: list, ori: list):
        """Fallback method for geometry without original data"""
        # Create transformation matrix for absolute world coordinates (fallback method)
        transform_matrix = np.eye(4)
        
        # Apply rotation (absolute orientation in world space)
        R_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(ori)
        transform_matrix[:3, :3] = R_matrix
        
        # Apply translation (absolute position in world space)
        transform_matrix[:3, 3] = pos
        
        # Use set_geometry_transform even for fallback
        scene.scene.set_geometry_transform(name, transform_matrix)
        self.__console.debug(f"Updated geometry {name} using fallback method with set_geometry_transform")

    def API_add_urdf(self, scene, name: str, robot, base_pos: list = [0, 0, 0], base_ori: list = [0, 0, 0], joint_config: dict = None):
        """Add URDF robot model to scene"""
        self.__console.debug(f"Call API_add_urdf : {name}")
        
        try:
            # Import required modules
            from scipy.spatial import transform as rotations
            from scipy.spatial import transform as transformations
            
            # Create base transformation matrix
            base_p = np.array(base_pos)
            base_R = rotations.Rotation.from_euler('xyz', base_ori).as_matrix()
            base_T = np.eye(4)
            base_T[:3, :3] = base_R
            base_T[:3, 3] = base_p
            
            # Use provided joint configuration or default to None (zero angles)
            cfg = joint_config if joint_config is not None else None
            
            # Compute forward kinematics
            fk = robot.visual_trimesh_fk(cfg=cfg)
            
            # Store original mesh data in identity transform state
            original_meshes = []
            for tm, T in fk.items():
                original_meshes.append(tm.copy())  # Store original mesh without any transform
            
            # Store URDF model info for potential updates
            self.__geometry_container[name] = {
                'type': 'urdf',
                'robot': robot,
                'base_transform': base_T,
                'joint_config': cfg if cfg else {},
                'mesh_count': len(fk),
                'original_meshes': original_meshes
            }
            
            # Add each mesh to the scene in identity state and use set_geometry_transform
            mesh_count = 1
            for tm, T in fk.items():
                # Convert trimesh to Open3D mesh WITHOUT applying any transform
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
                o3d_mesh.compute_vertex_normals()
                
                # Create material
                material = rendering.MaterialRecord()
                material.shader = "defaultLit"
                
                # Add to scene with unique name in identity state
                mesh_name = f"{name}_{mesh_count}"
                scene.scene.add_geometry(mesh_name, o3d_mesh, material)
                
                # Apply transform using set_geometry_transform
                final_transform = base_T @ T
                scene.scene.set_geometry_transform(mesh_name, final_transform)
                
                mesh_count += 1
            
            self.__console.debug(f"Added URDF {name} with {len(fk)} meshes to scene")
            
        except Exception as e:
            self.__console.error(f"Failed to add URDF {name}: {e}")

    def API_set_joint_angles(self, scene, name: str, joint_angles: list):
        """Set joint angles for URDF robot model and update visualization"""
        self.__console.debug(f"Call API_set_joint_angles : {name} with angles {joint_angles}")
        
        if name not in self.__geometry_container:
            self.__console.error(f"URDF model {name} not found in geometry container")
            return
        
        urdf_info = self.__geometry_container[name]
        if urdf_info.get('type') != 'urdf':
            self.__console.error(f"Geometry {name} is not a URDF model")
            return
        
        try:
            robot = urdf_info['robot']
            base_T = urdf_info['base_transform']
            
            # Get actuated joint names from robot
            joint_names = list(robot.actuated_joint_names)
            
            # Create joint configuration dictionary
            cfg = {}
            for i, joint_name in enumerate(joint_names):
                if i < len(joint_angles):
                    cfg[joint_name] = joint_angles[i]
                else:
                    cfg[joint_name] = 0.0  # Default to 0 if not enough angles provided
            
            # Update stored joint configuration
            urdf_info['joint_config'] = cfg
            
            # Compute forward kinematics with new joint angles
            fk = robot.visual_trimesh_fk(cfg=cfg)
            
            # Update meshes using efficient set_geometry_transform
            fk_list = list(fk.items())
            
            for i, (_, T) in enumerate(fk_list):
                mesh_name = f"{name}_{i+1}"
                
                # Calculate final transform matrix
                final_transform = base_T @ T
                
                # Use set_geometry_transform for efficient update
                try:
                    scene.scene.set_geometry_transform(mesh_name, final_transform)
                except Exception as ex:
                    self.__console.error(f"Failed to update transform for mesh {mesh_name}: {ex}")
            
            # Update mesh count in container
            urdf_info['mesh_count'] = len(fk)
            
            self.__console.debug(f"Updated URDF {name} with {len(fk)} meshes using joint angles {joint_angles}")
            
        except Exception as e:
            self.__console.error(f"Failed to set joint angles for URDF {name}: {e}")
    
        """Set individual joint value for URDF robot model and update visualization"""
        self.__console.debug(f"Call API_set_joint_value : joint={joint}, value={value}")
        
        # Find URDF model that contains this joint
        urdf_name = None
        urdf_info = None
        
        for name, info in self.__geometry_container.items():
            if info.get('type') == 'urdf':
                robot = info['robot']
                if joint in robot.actuated_joint_names:
                    urdf_name = name
                    urdf_info = info
                    break
        
        if urdf_name is None:
            self.__console.warning(f"Joint {joint} not found in any loaded URDF models")
            return
        
        try:
            robot = urdf_info['robot']
            base_T = urdf_info['base_transform']
            
            # Get current joint configuration or initialize if not exists
            current_cfg = urdf_info.get('joint_config', {})
            
            # Update the specific joint value
            current_cfg[joint] = value
            
            # Ensure all actuated joints have values (default to 0.0 if not set)
            for joint_name in robot.actuated_joint_names:
                if joint_name not in current_cfg:
                    current_cfg[joint_name] = 0.0
            
            # Update stored joint configuration
            urdf_info['joint_config'] = current_cfg
            
            # Compute forward kinematics with updated joint configuration
            fk = robot.visual_trimesh_fk(cfg=current_cfg)
            
            # Update meshes using efficient set_geometry_transform
            fk_list = list(fk.items())
            
            for i, (_, T) in enumerate(fk_list):
                mesh_name = f"{urdf_name}_{i+1}"
                
                # Calculate final transform matrix
                final_transform = base_T @ T
                
                # Use set_geometry_transform for efficient update
                try:
                    scene.scene.set_geometry_transform(mesh_name, final_transform)
                except Exception as ex:
                    self.__console.error(f"Failed to update transform for mesh {mesh_name}: {ex}")
            
            # Update mesh count in container
            urdf_info['mesh_count'] = len(fk)
            
            self.__console.debug(f"Updated URDF {urdf_name} joint {joint} to {value} radians")
            
        except Exception as e:
            self.__console.error(f"Failed to set joint value for {joint}: {e}")
    
    def API_set_joint_angle(self, scene, joint: str, value: float):
        """Set individual joint angle by joint name and update URDF visualization
        
        Args:
            scene: The Open3D scene to update
            joint: Name of the joint as defined in URDF
            value: Rotation value in radians
        """
        self.__console.debug(f"Call API_set_joint_angle : joint={joint}, value={value}")
        
        # Find URDF model that contains this joint
        urdf_name = None
        urdf_info = None
        
        for model_name, info in self.__geometry_container.items():
            if info.get('type') == 'urdf':
                robot = info['robot']
                if joint in robot.actuated_joint_names:
                    urdf_name = model_name
                    urdf_info = info
                    break
        
        if urdf_name is None:
            self.__console.warning(f"Joint {joint} not found in any loaded URDF models")
            return
        
        try:
            robot = urdf_info['robot']
            base_T = urdf_info['base_transform']
            
            # Get current joint configuration or initialize if not exists
            current_cfg = urdf_info.get('joint_config', {})
            
            # Update the specific joint value
            current_cfg[joint] = value
            
            # Ensure all actuated joints have values (default to 0.0 if not set)
            for joint_name in robot.actuated_joint_names:
                if joint_name not in current_cfg:
                    current_cfg[joint_name] = 0.0
            
            # Update stored joint configuration
            urdf_info['joint_config'] = current_cfg
            
            # Compute forward kinematics with updated joint configuration
            fk = robot.visual_trimesh_fk(cfg=current_cfg)
            
            # Update meshes using efficient set_geometry_transform
            fk_list = list(fk.items())
            
            for i, (_, T) in enumerate(fk_list):
                mesh_name = f"{urdf_name}_{i+1}"
                
                # Calculate final transform matrix
                final_transform = base_T @ T
                
                # Use set_geometry_transform for efficient update
                try:
                    scene.scene.set_geometry_transform(mesh_name, final_transform)
                except Exception as ex:
                    self.__console.error(f"Failed to update transform for mesh {mesh_name}: {ex}")
            
            self.__console.debug(f"Updated URDF {urdf_name} joint {joint} to {value} radians")
            
        except Exception as e:
            self.__console.error(f"Failed to set joint angle for {joint}: {e}")

    def _remove_urdf_geometry(self, scene, name: str):
        """Remove all geometry associated with a URDF model"""
        if name not in self.__geometry_container:
            return
        
        urdf_info = self.__geometry_container[name]
        mesh_count = urdf_info.get('mesh_count', 0)
        
        # Remove all meshes associated with this URDF
        for i in range(1, mesh_count + 1):
            mesh_name = f"{name}_{i}"
            try:
                scene.scene.remove_geometry(mesh_name)
            except:
                # Mesh might not exist, continue
                pass

    
