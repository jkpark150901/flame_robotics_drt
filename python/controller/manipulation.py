"""
Robot Manipulation Module for Forward Kinematics Calculation
Provides forward kinematics computation for robot manipulators based on URDF configuration
@author: Byunghun Hwang (bh.hwang@iae.re.kr)
"""

import os
import numpy as np
from scipy.spatial import transform as rotations
try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
    print(f"Using pinocchio version {pin.__version__} for kinematics calculations")
except ImportError:
    print("pinocchio not available, falling back to URDF parser")
    print("To install pinocchio: pip install pin")
    PINOCCHIO_AVAILABLE = False

if not PINOCCHIO_AVAILABLE:
    from urdf_parser import URDF
    
from util.logger.console import ConsoleLogger


class Kinematics:
    """Module for robot manipulator forward kinematics calculations"""
    
    def __init__(self, config: dict):
        """
        Initialize manipulation module with configuration
        
        Args:
            config: Configuration dictionary containing URDF robot definitions
        """
        self.__console = ConsoleLogger.get_logger()
        self.__config = config
        self.__robots = {}  # Dict to store loaded robot models
        self.__joint_configs = {}  # Dict to store current joint configurations
        
        # Load robot models from config
        self._load_robots_from_config()
        
    def _load_robots_from_config(self):
        """Load robot models from configuration file"""
        if "urdf" not in self.__config:
            self.__console.warning("No URDF configuration found in config")
            return
            
        for urdf_config in self.__config["urdf"]:
            try:
                robot_name = urdf_config["name"]
                urdf_path = urdf_config["path"]
                base_transform = urdf_config.get("base", [0, 0, 0, 0, 0, 0])
                
                # Get absolute path
                if not os.path.isabs(urdf_path):
                    # Try multiple possible root paths
                    possible_roots = [
                        self.__config.get("root_path", ""),
                        os.getcwd(),
                        os.path.join(os.getcwd(), ".."),  # Go up one directory
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go to project root
                    ]
                    
                    for root_path in possible_roots:
                        if root_path:
                            test_path = os.path.join(root_path, urdf_path)
                            if os.path.isfile(test_path):
                                urdf_path = test_path
                                break
                    
                    # If still not found, just use the original path
                    if not os.path.isfile(urdf_path):
                        urdf_path = os.path.join(os.getcwd(), urdf_path)
                
                # Create base transformation matrix
                base_pos = np.array(base_transform[0:3])
                base_ori = np.deg2rad(np.array(base_transform[3:6]))  # Convert degrees to radians
                
                base_R = rotations.Rotation.from_euler('xyz', base_ori).as_matrix()
                base_T = np.eye(4)
                base_T[:3, :3] = base_R
                base_T[:3, 3] = base_pos
                
                # Try to load with pinocchio first if available
                pin_success = False
                if PINOCCHIO_AVAILABLE:
                    try:
                        # Load robot model with pinocchio
                        pin_model = pin.buildModelFromUrdf(urdf_path)
                        pin_data = pin_model.createData()
                        
                        # Get joint names (excluding universe joint)
                        joint_names = []
                        for i in range(1, pin_model.njoints):  # Skip universe joint (index 0)
                            joint_name = pin_model.names[i]
                            joint_names.append(joint_name)
                        
                        # Store robot information with pinocchio
                        self.__robots[robot_name] = {
                            'pin_model': pin_model,
                            'pin_data': pin_data,
                            'type': 'pinocchio',
                            'base_transform': base_T,
                            'joint_names': joint_names,
                            'urdf_path': urdf_path
                        }
                        
                        self.__console.debug(f"Loaded robot {robot_name} with pinocchio - {len(joint_names)} joints: {joint_names}")
                        pin_success = True
                        
                    except Exception as e:
                        self.__console.error(f"Failed to load {robot_name} with pinocchio: {e}")
                        self.__console.info(f"Falling back to URDF parser for {robot_name}")
                
                # Fallback to URDF parser if pinocchio failed or not available
                if not pin_success:
                    robot = URDF.load(urdf_path, lazy_load_meshes=True)
                    
                    # Store robot information with URDF parser
                    self.__robots[robot_name] = {
                        'robot': robot,
                        'type': 'urdf',
                        'base_transform': base_T,
                        'joint_names': list(robot.actuated_joint_names),
                        'urdf_path': urdf_path
                    }
                    
                    self.__console.debug(f"Loaded robot {robot_name} with URDF parser - {len(robot.actuated_joint_names)} joints")
                
                # Initialize joint configuration to zero
                robot_info = self.__robots[robot_name]
                self.__joint_configs[robot_name] = {
                    joint_name: 0.0 for joint_name in robot_info['joint_names']
                }
                
                self.__console.debug(f"Loaded robot {robot_name} with {len(self.__joint_configs[robot_name])} joints")
                
            except Exception as e:
                self.__console.error(f"Failed to load robot {urdf_config.get('name', 'unknown')}: {e}")
    
    def get_robot_names(self):
        """Get list of available robot names"""
        return list(self.__robots.keys())
    
    def get_joint_names(self, robot_name: str):
        """Get joint names for a specific robot"""
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return []
        return self.__robots[robot_name]['joint_names'].copy()
    
    def set_joint_angles(self, robot_name: str, joint_angles: dict):
        """
        Set joint angles for a robot
        
        Args:
            robot_name: Name of the robot
            joint_angles: Dictionary with joint_name: angle_in_radians pairs
        """
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return False
            
        try:
            # Update joint configuration
            for joint_name, angle in joint_angles.items():
                if joint_name in self.__joint_configs[robot_name]:
                    self.__joint_configs[robot_name][joint_name] = float(angle)
                else:
                    self.__console.warning(f"Joint {joint_name} not found in robot {robot_name}")
            
            return True
            
        except Exception as e:
            self.__console.error(f"Failed to set joint angles for {robot_name}: {e}")
            return False
    
    def set_joint_angle(self, robot_name: str, joint_name: str, angle: float):
        """
        Set angle for a specific joint
        
        Args:
            robot_name: Name of the robot
            joint_name: Name of the joint
            angle: Joint angle in radians
        """
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found. Available robots: {list(self.__robots.keys())}")
            return False
            
        if joint_name not in self.__joint_configs[robot_name]:
            available_joints = list(self.__joint_configs[robot_name].keys())
            self.__console.error(f"Joint {joint_name} not found in robot {robot_name}. Available joints: {available_joints}")
            return False
            
        try:
            self.__joint_configs[robot_name][joint_name] = float(angle)
            self.__console.debug(f"Set {robot_name}.{joint_name} = {angle:.4f} rad")
            return True
        except Exception as e:
            self.__console.error(f"Failed to set joint angle for {robot_name}.{joint_name}: {e}")
            return False
    
    def get_joint_angles(self, robot_name: str):
        """Get current joint angles for a robot"""
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return {}
        return self.__joint_configs[robot_name].copy()
    
    def compute_fk(self, robot_name: str, joint_config: dict = None):
        """
        Compute forward kinematics for end-effector position and orientation
        
        Args:
            robot_name: Name of the robot
            joint_config: Optional joint configuration. If None, uses current stored configuration
            
        Returns:
            dict: {'position': [x, y, z], 'orientation': [rx, ry, rz], 'transform': 4x4_matrix}
            Returns None if computation fails
        """
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return None
            
        try:
            robot_info = self.__robots[robot_name]
            base_T = robot_info['base_transform']
            
            # Use provided joint config or current stored config
            if joint_config is None:
                joint_config = self.__joint_configs[robot_name]
            
            if robot_info['type'] == 'pinocchio':
                # Use pinocchio for forward kinematics
                pin_model = robot_info['pin_model']
                pin_data = robot_info['pin_data']
                
                # Create joint angle vector for pinocchio
                q = np.zeros(pin_model.nq)  # Configuration vector
                joint_names = robot_info['joint_names']
                
                # joint_names = 액추에이트 관절(universe 제외)이고 nq도 그 수와 같으므로
                # q[i] = joint_names[i] (오프셋 없음)
                for i, joint_name in enumerate(joint_names):
                    if i < len(q):
                        q[i] = joint_config.get(joint_name, 0.0)

                # Compute forward kinematics
                pin.forwardKinematics(pin_model, pin_data, q)
                pin.updateFramePlacements(pin_model, pin_data)
                
                # Get end-effector transform (last frame)
                end_effector_frame_id = pin_model.nframes - 1
                end_effector_T = pin_data.oMf[end_effector_frame_id].homogeneous
                
            else:
                # Use URDF parser for forward kinematics
                robot = robot_info['robot']
                
                # Ensure all joints have values
                cfg = {}
                for joint_name in robot_info['joint_names']:
                    cfg[joint_name] = joint_config.get(joint_name, 0.0)
                
                # Compute forward kinematics using URDF
                fk_result = robot.link_fk(cfg=cfg)
                
                # Get end-effector transform (typically the last link)
                end_effector_links = list(fk_result.keys())
                if not end_effector_links:
                    self.__console.error(f"No links found for robot {robot_name}")
                    return None
                    
                # Use the last link as end-effector
                end_effector_link = end_effector_links[-1]
                end_effector_T = fk_result[end_effector_link]
            
            # Apply base transformation
            final_T = base_T @ end_effector_T
            
            # Extract position and orientation
            position = final_T[:3, 3].tolist()
            rotation_matrix = final_T[:3, :3]
            
            # Convert rotation matrix to Euler angles (XYZ convention)
            orientation_rad = rotations.Rotation.from_matrix(rotation_matrix).as_euler('xyz')
            orientation = orientation_rad.tolist()
            
            result = {
                'position': position,
                'orientation': orientation,  # in radians
                'orientation_deg': np.rad2deg(orientation_rad).tolist(),  # in degrees for convenience
                'transform': final_T.tolist()
            }
            
            # self.__console.debug(f"FK for {robot_name}: pos={position}, ori_deg={np.rad2deg(orientation_rad)}")
            return result
            
        except Exception as e:
            self.__console.error(f"Failed to compute forward kinematics for {robot_name}: {e}")
            return None
    
    def compute_ik(self, robot_name: str, target_position: list, target_orientation: list = None, 
                                 initial_joint_config: dict = None, max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Compute inverse kinematics to reach target pose
        
        Args:
            robot_name: Name of the robot
            target_position: Target position [x, y, z]
            target_orientation: Target orientation [rx, ry, rz] in radians (optional)
            initial_joint_config: Initial joint configuration for IK solver
            max_iterations: Maximum iterations for IK solver
            tolerance: Convergence tolerance
            
        Returns:
            dict: {'joint_angles': {joint_name: angle}, 'success': bool, 'error': float}
            Returns None if robot not found or IK fails
        """
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return None
            
        robot_info = self.__robots[robot_name]
        
        # Only pinocchio supports IK directly
        if robot_info['type'] != 'pinocchio':
            self.__console.warning(f"Inverse kinematics only available with pinocchio backend")
            return None
            
        try:
            pin_model = robot_info['pin_model']
            pin_data = robot_info['pin_data']
            base_T = robot_info['base_transform']
            
            # Create target transform matrix
            target_T = np.eye(4)
            target_T[:3, 3] = target_position
            
            if target_orientation is not None:
                target_R = rotations.Rotation.from_euler('xyz', target_orientation).as_matrix()
                target_T[:3, :3] = target_R
            else:
                # If no orientation specified, use current orientation
                current_fk = self.compute_fk(robot_name)
                if current_fk:
                    current_T = np.array(current_fk['transform'])
                    target_T[:3, :3] = current_T[:3, :3]
            
            # Apply inverse base transform
            target_T_local = np.linalg.inv(base_T) @ target_T
            
            # Initial joint configuration
            if initial_joint_config is None:
                initial_joint_config = self.__joint_configs[robot_name]
            
            # Create initial configuration vector
            q_init = np.zeros(pin_model.nq)
            joint_names = robot_info['joint_names']

            for i, joint_name in enumerate(joint_names):
                if i < len(q_init):
                    q_init[i] = initial_joint_config.get(joint_name, 0.0)
            
            # Get end-effector frame ID
            end_effector_frame_id = pin_model.nframes - 1
            
            # Create target SE3 object
            target_se3 = pin.SE3(target_T_local[:3, :3], target_T_local[:3, 3])
            
            # Solve IK using CLIK (Closed-Loop Inverse Kinematics)
            q_result = q_init.copy()
            error = float('inf')
            
            for iteration in range(max_iterations):
                # Compute forward kinematics
                pin.forwardKinematics(pin_model, pin_data, q_result)
                pin.updateFramePlacements(pin_model, pin_data)
                
                # Get current end-effector pose
                current_pose = pin_data.oMf[end_effector_frame_id]
                
                # Compute error
                pose_error = pin.log6(current_pose.inverse() * target_se3)
                error = np.linalg.norm(pose_error.vector)
                
                print(f"Iteration {iteration+1}: error={error:.6f}")
                
                if error < tolerance:
                    break
                
                # Compute Jacobian
                pin.computeFrameJacobian(pin_model, pin_data, q_result, end_effector_frame_id)
                J = pin_data.J
                
                # Damped least squares solution
                lambda_damping = 0.01  # Damping parameter
                J_damped = J.T @ np.linalg.inv(J @ J.T + lambda_damping**2 * np.eye(J.shape[0]))
                dq = J_damped @ pose_error.vector
                
                # Update configuration
                q_result = pin.integrate(pin_model, q_result, dq)
            
            # Extract joint angles from result
            joint_angles = {}
            for i, joint_name in enumerate(joint_names):
                if i + 1 < len(q_result):
                    joint_angles[joint_name] = q_result[i + 1]
            
            success = error < tolerance
            
            self.__console.debug(f"IK for {robot_name}: iterations={iteration+1}, error={error:.6f}, success={success}")
            
            return {
                'joint_angles': joint_angles,
                'success': success,
                'error': error,
                'iterations': iteration + 1
            }
            
        except Exception as e:
            self.__console.error(f"Failed to compute inverse kinematics for {robot_name}: {e}")
            return None
    
    def get_end_effector_pose(self, robot_name: str):
        """
        Get current end-effector pose using stored joint configuration
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            dict: {'position': [x, y, z], 'orientation': [rx, ry, rz]} or None if failed
        """
        fk_result = self.compute_fk(robot_name)
        if fk_result:
            return {
                'position': fk_result['position'],
                'orientation': fk_result['orientation']
            }
        return None
    
    def reset_joint_angles(self, robot_name: str = None):
        """
        Reset joint angles to zero
        
        Args:
            robot_name: Name of the robot. If None, resets all robots
        """
        if robot_name:
            if robot_name in self.__joint_configs:
                for joint_name in self.__joint_configs[robot_name]:
                    self.__joint_configs[robot_name][joint_name] = 0.0
                self.__console.debug(f"Reset joint angles for robot {robot_name}")
            else:
                self.__console.error(f"Robot {robot_name} not found")
        else:
            for name in self.__joint_configs:
                for joint_name in self.__joint_configs[name]:
                    self.__joint_configs[name][joint_name] = 0.0
            self.__console.debug("Reset joint angles for all robots")
    
    def get_robot_info(self, robot_name: str):
        """
        Get information about a robot
        
        Returns:
            dict: Robot information including joint names, base transform, etc.
        """
        if robot_name not in self.__robots:
            return None
            
        robot_info = self.__robots[robot_name].copy()
        # Don't return the actual robot object for safety
        result = {
            'joint_names': robot_info['joint_names'],
            'base_transform': robot_info['base_transform'].tolist(),
            'urdf_path': robot_info['urdf_path'],
            'current_joint_config': self.__joint_configs[robot_name].copy()
        }
        return result
    
    def compute_all_robots_fk(self):
        """
        Compute forward kinematics for all robots
        
        Returns:
            dict: {robot_name: fk_result} for all robots
        """
        results = {}
        for robot_name in self.__robots:
            fk_result = self.compute_fk(robot_name)
            if fk_result:
                results[robot_name] = fk_result
        return results
    
    def get_joint_limits(self, robot_name: str):
        """
        Get joint limits for specified robot
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            dict: Dictionary with joint names as keys and {'lower': float, 'upper': float} as values
        """
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return {}
            
        if PINOCCHIO_AVAILABLE:
            robot_data = self.__robots[robot_name]
            pin_model = robot_data['pin_model']
            joint_names = robot_data['joint_names']
            joint_limits = {}
            
            for i, joint_name in enumerate(joint_names):
                # pinocchio joint indices start from 1 (0 is universe joint)
                joint_idx = i + 1
                if joint_idx < pin_model.nq:
                    joint_limits[joint_name] = {
                        'lower': float(pin_model.lowerPositionLimit[joint_idx]),
                        'upper': float(pin_model.upperPositionLimit[joint_idx])
                    }
            
            return joint_limits
        else:
            # Fallback to URDF parser method
            robot_data = self.__robots[robot_name]
            urdf_model = robot_data['urdf_model']
            joint_names = robot_data['joint_names']
            joint_limits = {}
            
            for joint_name in joint_names:
                joint = urdf_model.joint_map.get(joint_name)
                if joint and joint.limit:
                    joint_limits[joint_name] = {
                        'lower': joint.limit.lower,
                        'upper': joint.limit.upper
                    }
                else:
                    # Default limits if not specified
                    joint_limits[joint_name] = {
                        'lower': -3.14159,
                        'upper': 3.14159
                    }
            
            return joint_limits

    def compute_ik_with_callback(self, robot_name: str, target_position: list, target_orientation: list = None, 
                               iteration_callback=None, initial_joint_config: dict = None, 
                               max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Compute inverse kinematics with callback function for iteration updates
        
        Args:
            robot_name: Name of the robot
            target_position: Target position [x, y, z]
            target_orientation: Target orientation [rx, ry, rz] in radians (optional)
            iteration_callback: Function called at each iteration with (iteration, joint_angles, error)
            initial_joint_config: Initial joint configuration for IK solver
            max_iterations: Maximum iterations for IK solver
            tolerance: Convergence tolerance
            
        Returns:
            dict: {'joint_angles': {joint_name: angle}, 'success': bool, 'error': float, 'iterations': int}
            Returns None if robot not found or IK fails
        """
        if robot_name not in self.__robots:
            self.__console.error(f"Robot {robot_name} not found")
            return None
            
        robot_info = self.__robots[robot_name]
        
        # Only pinocchio supports IK directly
        if robot_info['type'] != 'pinocchio':
            self.__console.warning(f"Inverse kinematics only available with pinocchio backend")
            return None
            
        try:
            pin_model = robot_info['pin_model']
            pin_data = robot_info['pin_data']
            base_T = robot_info['base_transform']
            
            # Create target transform matrix
            target_T = np.eye(4)
            target_T[:3, 3] = target_position
            
            if target_orientation is not None:
                target_R = rotations.Rotation.from_euler('xyz', target_orientation).as_matrix()
                target_T[:3, :3] = target_R
            else:
                # If no orientation specified, use current orientation
                current_fk = self.compute_fk(robot_name)
                if current_fk:
                    current_T = np.array(current_fk['transform'])
                    target_T[:3, :3] = current_T[:3, :3]
            
            # Apply inverse base transform
            target_T_local = np.linalg.inv(base_T) @ target_T
            
            # Debug: Print initial and target information for callback version
            self.__console.info(f"IK Setup for {robot_name} (with callback):")
            self.__console.info(f"Target position (global): [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
            self.__console.info(f"Target position (local):  [{target_T_local[0,3]:.3f}, {target_T_local[1,3]:.3f}, {target_T_local[2,3]:.3f}]")
            if target_orientation is not None:
                self.__console.info(f"Target orientation: [{target_orientation[0]:.3f}, {target_orientation[1]:.3f}, {target_orientation[2]:.3f}] rad")
            
            # Basic workspace check - rough estimation
            target_distance = np.linalg.norm(target_T_local[:3, 3])
            estimated_reach = 2.0  # Rough estimate for typical 6DOF arm reach (meters)
            
            if target_distance > estimated_reach:
                self.__console.warning(f"Target may be outside workspace: distance={target_distance:.3f}m > estimated_reach={estimated_reach:.3f}m")
                self.__console.info("Consider using a closer target position")
            else:
                self.__console.info(f"Target within estimated workspace: distance={target_distance:.3f}m")
            
            # Check current end-effector position before starting
            initial_fk = self.compute_fk(robot_name)
            if initial_fk:
                current_pos = initial_fk['position']
                initial_error = np.linalg.norm(np.array(current_pos) - np.array(target_position))
                self.__console.info(f"Initial end-effector position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                self.__console.info(f"Initial position error: {initial_error:.6f} m")
            
            # Initial joint configuration
            if initial_joint_config is None:
                initial_joint_config = self.__joint_configs[robot_name]
            
            # Create initial configuration vector
            q_init = np.zeros(pin_model.nq)
            joint_names = robot_info['joint_names']
            
            # Debug: Print initial joint configuration
            self.__console.info(f"Initial joint configuration for {robot_name}:")
            for i, joint_name in enumerate(joint_names):
                if i + 1 < len(q_init):
                    joint_value = initial_joint_config.get(joint_name, 0.0)
                    q_init[i + 1] = joint_value
                    self.__console.info(f"  {joint_name}: {joint_value:.4f} rad ({np.rad2deg(joint_value):.1f}°)")
                    
            # Print q_init vector
            self.__console.info(f"q_init vector (size {len(q_init)}): {q_init[:min(10, len(q_init))]}")  # First 10 elements only
            
            # Get end-effector frame ID
            end_effector_frame_id = pin_model.nframes - 1
            
            # Create target SE3 object
            target_se3 = pin.SE3(target_T_local[:3, :3], target_T_local[:3, 3])
            
            # Early termination if target is too far
            if target_distance > estimated_reach * 1.2:  # Allow some margin
                self.__console.error(f"Target definitely outside workspace, aborting IK")
                return {
                    'joint_angles': {},
                    'success': False,
                    'error': target_distance,
                    'iterations': 0
                }
            
            # Solve IK using CLIK (Closed-Loop Inverse Kinematics)
            q_result = q_init.copy()
            error = float('inf')
            prev_error = float('inf')
            stagnation_count = 0
            prev_dq = None
            
            for iteration in range(max_iterations):
                # Compute forward kinematics
                pin.forwardKinematics(pin_model, pin_data, q_result)
                pin.updateFramePlacements(pin_model, pin_data)
                
                # Get current end-effector pose
                current_pose = pin_data.oMf[end_effector_frame_id]
                
                # Compute error
                pose_error = pin.log6(current_pose.inverse() * target_se3)
                error = np.linalg.norm(pose_error.vector)
                
                # Debug: Print detailed information
                current_position = current_pose.translation
                target_position_local = target_T_local[:3, 3]
                position_error = np.linalg.norm(current_position - target_position_local)
                
                self.__console.debug(f"IK Debug [{iteration}]: error={error:.6f}, pos_error={position_error:.6f}")
                self.__console.debug(f"Current pos: [{current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f}]")
                self.__console.debug(f"Target pos:  [{target_position_local[0]:.3f}, {target_position_local[1]:.3f}, {target_position_local[2]:.3f}]")
                self.__console.debug(f"Pose error vector: {pose_error.vector[:3]}")
                
                # Extract current joint angles for callback
                current_joint_angles = {}
                for i, joint_name in enumerate(joint_names):
                    if i + 1 < len(q_result):
                        current_joint_angles[joint_name] = q_result[i + 1]
                
                # Call iteration callback if provided
                if iteration_callback:
                    should_continue = iteration_callback(iteration, current_joint_angles, error)
                    if should_continue is False:
                        self.__console.info("IK computation stopped by callback")
                        break
                
                if error < tolerance:
                    break
                
                # Compute Jacobian - Use the correct reference frame
                # For end-effector frame Jacobian in local coordinates
                J = pin.computeFrameJacobian(pin_model, pin_data, q_result, end_effector_frame_id, pin.LOCAL)
                
                # Debug: Check Jacobian
                self.__console.debug(f"Jacobian shape: {J.shape}, rank: {np.linalg.matrix_rank(J)}")
                self.__console.debug(f"Jacobian condition number: {np.linalg.cond(J @ J.T):.2e}")
                
                # Check if Jacobian is degenerate
                if np.linalg.matrix_rank(J) < min(J.shape):
                    self.__console.warning(f"Jacobian is rank deficient: rank={np.linalg.matrix_rank(J)}, expected={min(J.shape)}")
                
                # Adaptive damping based on error magnitude
                base_damping = 0.1
                adaptive_damping = base_damping + 0.5 * min(error, 1.0)  # Increase damping for large errors
                
                JJT = J @ J.T
                JJT_damped = JJT + adaptive_damping**2 * np.eye(JJT.shape[0])
                
                # Use pseudo-inverse for better numerical stability
                try:
                    J_pinv = J.T @ np.linalg.inv(JJT_damped)
                    dq = J_pinv @ pose_error.vector
                except np.linalg.LinAlgError:
                    self.__console.error("Singular matrix in Jacobian inversion")
                    break
                
                # Step size control to prevent oscillations
                max_step_size = 0.5  # Maximum joint angle change per iteration (radians)
                dq_norm = np.linalg.norm(dq)
                self.__console.debug(f"Joint velocity norm: {dq_norm:.6f}, adaptive damping: {adaptive_damping:.3f}")
                
                if dq_norm > max_step_size:
                    # Scale down the step to prevent large jumps
                    dq = dq * (max_step_size / dq_norm)
                    self.__console.debug(f"Step size limited: scaled to {np.linalg.norm(dq):.6f}")
                
                if dq_norm < 1e-8:
                    self.__console.warning("Joint velocity too small - possible convergence or singularity")
                    # Try with reduced damping
                    if adaptive_damping > 0.01:
                        adaptive_damping *= 0.5
                        self.__console.info(f"Reducing damping to {adaptive_damping:.3f}")
                        continue
                    else:
                        break
                
                # Momentum-based update to reduce oscillations
                if iteration > 0:
                    momentum = 0.1  # Momentum factor
                    if 'prev_dq' in locals():
                        dq = (1 - momentum) * dq + momentum * prev_dq
                
                prev_dq = dq.copy()  # Store for next iteration
                
                # Check if we're making progress
                if iteration > 0 and abs(error - prev_error) < tolerance * 0.01:
                    stagnation_count = getattr(locals(), 'stagnation_count', 0) + 1
                    if stagnation_count > 5:
                        self.__console.warning("IK stagnating - stopping early")
                        break
                else:
                    stagnation_count = 0
                
                prev_error = error
                
                # Update configuration
                q_result = pin.integrate(pin_model, q_result, dq)
            
            # Extract joint angles from result
            joint_angles = {}
            for i, joint_name in enumerate(joint_names):
                if i + 1 < len(q_result):
                    joint_angles[joint_name] = q_result[i + 1]
            
            success = error < tolerance
            
            self.__console.debug(f"IK with callback for {robot_name}: iterations={iteration+1}, error={error:.6f}, success={success}")
            
            return {
                'joint_angles': joint_angles,
                'success': success,
                'error': error,
                'iterations': iteration + 1
            }
            
        except Exception as e:
            self.__console.error(f"Failed to compute inverse kinematics with callback for {robot_name}: {e}")
            return None