'''
DRT Monitor Application Window class
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

try:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox, QDialog
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal, QRegularExpression
except ImportError:
    print("PyQt6 is required to run this application.")

import zmq
import os, sys
import pathlib
import json
import threading
import re
import numpy as np
import math
from functools import partial

from common.zpipe import AsyncZSocket, ZPipe
from util.logger.console import ConsoleLogger
from .geometry_model import GeometryTableModel  # geometry model handling
from .tcpview_model import TCPViewTableModel  # tcp view model handling
from .manipulation import Kinematics  # kinematics solver for FK/IK calculation

""" Global variables """
# DO not change this value unless necessary
JOINT_ANGLE_SCALE_FACTOR = 1000

# Do change these values according to your robot configuration
DDA_SIDE_MANIPULATOR = "dda_rb10_1300e"
RT_SIDE_MANIPULATOR = "rt_rb10_1300e"


class AppWindow(QMainWindow):
    def __init__(self, config:dict, zpipe:ZPipe):
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger()
        self.__config = config

        # Initialize table models
        self.__geometry_model = GeometryTableModel()
        self.__tcpview_model = TCPViewTableModel(config=self.__config)
        
        # Initialize kinematics solver
        self.__kinematics = Kinematics(config)
        
        # Dictionary to map joint names to their UI controls
        self.__joint_ui_mapping = {}
        
        # Flag to stop IK computation
        self.__ik_stop_flag = False

        try:            
            if "gui" in config:
                # load UI
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                    self.setWindowTitle(config.get("window_title", "DRT Control Window"))

                    # Create socket for communication (as server)
                    self.__socket = AsyncZSocket("Controller", "publish")
                    if self.__socket.create(pipeline=zpipe):
                        transport = config.get("transport", "tcp")
                        port = config.get("port", 9001)
                        host = config.get("host", "localhost")
                        if self.__socket.join(transport, host, port):
                            self.__socket.set_message_callback(self.__on_data_received)
                            self.__console.debug(f"Socket created : {transport}://{host}:{port}")
                        else:
                            self.__console.error("Failed to join AsyncZsocket")
                    else:
                        self.__console.error("Failed to create AsyncZsocket")

                    # Initialize URDF parser and joint limits
                    self.urdf_parser = None
                    _rt_joint_limits = {}
                    _dda_joint_limits = {}
                    
                    # Get joint limits from manipulation module
                    for robot_name in self.__kinematics.get_robot_names():
                        joint_limits = self.__kinematics.get_joint_limits(robot_name)
                        if DDA_SIDE_MANIPULATOR in robot_name:
                            _dda_joint_limits = joint_limits
                            self.__console.info(f"Found {len(_dda_joint_limits)} joints with limits for DDA-side Manipulator")
                        elif RT_SIDE_MANIPULATOR in robot_name:
                            _rt_joint_limits = joint_limits
                            self.__console.info(f"Found {len(_rt_joint_limits)} joints with limits for RT-side Manipulator")

                    # menu actions
                    self.actionUtilGeneratePCD.triggered.connect(self.on_select_generate_pcd)

                    # Initialize sliders with URDF joint limits
                    self._setup_sliders(dda_side_joint_limits=_dda_joint_limits, rt_side_joint_limits=_rt_joint_limits)

                    # Setup geometry table
                    self.table_geometry.setModel(self.__geometry_model)
                    self.__geometry_model.geometryTransformChanged.connect(self.on_geometry_transform_changed)

                    # Setup TCP view table
                    self.table_tcp_move.setModel(self.__tcpview_model)
                    
                    # Enable delete key functionality for geometry table
                    self.table_geometry.keyPressEvent = self.on_geometry_table_key_press
                    
                    # Button events
                    self.btn_open_pcd.clicked.connect(self.on_open_pcd)
                    self.btn_load_pcd.clicked.connect(self.on_load_pcd)
                    self.btn_run_simulation.clicked.connect(self.on_run_simulation)
                    self.btn_stop_simulation.clicked.connect(self.on_stop_simulation)
                    self.btn_geometry_remove_all.clicked.connect(self.on_btn_geometry_remove_all)
                    self.btn_tcp_move.clicked.connect(self.on_tcp_move)
                    self.btn_tcp_movestop.clicked.connect(self.on_tcp_movestop)
                    self.btn_set_initial_pose.clicked.connect(self.on_set_initial_pose)

                    # Initialize TCP table with initial robot poses (all joints at 0)
                    self._initialize_tcp_table()

                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
                
        except Exception as e:
            self.__console.error(f"{e}")

    def __on_data_received(self, multipart_data):
        """Callback function for zpipe data reception"""
        try:
            if len(multipart_data) >= 2:
                topic = multipart_data[0]
                msg = multipart_data[1]
                self.__call(topic, msg)
        except Exception as e:
            self.__console.error(f"({self.__class__.__name__}) Error processing received data: {e}")
    
    def closeEvent(self, event:QCloseEvent) -> None:
        """ Handle close event """
        try:
            # Stop any ongoing IK computation first
            self.__ik_stop_flag = True
            self.__console.info("Stopping any ongoing IK computations")
            
            # Clear all geometry in viewer3d before closing
            self.__console.info("Clearing all geometry before closing controller window")
            self.__call(socket=self.__socket, function="API_clear_all_geometry", kwargs={})

            # Clear geometry table model
            self.__geometry_model.clear_all_geometry()

            # Clean up subscriber socket first
            if hasattr(self, '_AppWindow__socket') and self.__socket:
                self.__socket.destroy_socket()
                self.__console.debug(f"({self.__class__.__name__}) Destroyed socket")

        except Exception as e:
            self.__console.error(f"Error during window close: {e}")
        finally:
            self.__console.info("Successfully Closed")
            return super().closeEvent(event)
    
    def on_open_pcd(self):
        """ Open PCD file dialog """
        pcd_file, _ = QFileDialog.getOpenFileName(self, "Open PCD File", "", "PCD Files (*.pcd);PLY Files (*.ply);All Files (*)")
        if pcd_file:
            self.__console.info(f"Selected PCD file: {pcd_file}")
            self.edit_pcd_file.setText(pcd_file)
        else:
            self.__console.warning("No PCD file selected.")
    
    def on_load_pcd(self):
        """ Load PCD file """
        self.__console.info(f"({self.__class__.__name__}) Loading PCD file")
        pcd_file = self.edit_pcd_file.text()
        pcd_name = os.path.basename(pcd_file)
        
        # Add to geometry table
        self.__geometry_model.add_geometry(pcd_name, pos=[0.0, 0.0, 0.0], ori=[0.0, 0.0, 0.0], geometry_type="pcd")
        
        # Send to visualizer
        self.__call(socket=self.__socket, function="API_add_pcd", kwargs={"name":pcd_name, "path":pcd_file, "pos":[0.0, 0.0, 0.0], "point_size":0.2})
    

    def on_run_simulation(self):
        """ Run Simulation """
        self.__console.info(f"({self.__class__.__name__}) Running Simulation")
        

    def on_stop_simulation(self):
        """ Stop Simulation """
        self.__console.info(f"({self.__class__.__name__}) Stop Simulation")

    def on_btn_geometry_remove_all(self):
        """ Clear all geometry """ 
        
        self.__console.info(f"({self.__class__.__name__}) Clear all geometry")
        
        # Clear geometry table
        self.__geometry_model.clear_all_geometry()
        
        # Send to visualizer
        self.__call(socket=self.__socket, function="API_clear_all_geometry", kwargs={})

    def on_tcp_move(self):
        """Read TCP table values and perform inverse kinematics"""
        try:
            # Reset stop flag when starting new IK computation
            self.__ik_stop_flag = False
            
            # Get all TCP data from the table model
            tcp_data = self.__tcpview_model.get_all_tcp_data()
            
            if not tcp_data:
                self.__console.warning("No TCP data available in table")
                return
            
            # # Process each robot's TCP target
            for robot_name, data in tcp_data.items():
                if self.__ik_stop_flag:
                    self.__console.info("IK computation stopped by user request")
                    break
                    
                target_position = data['pos']  # [x, y, z]
                target_orientation = data['ori']  # [rx, ry, rz] in radians
                
                self.__console.info(f"Performing IK for {robot_name}: pos={target_position}, ori={target_orientation}")
                
                # Perform inverse kinematics with iteration-based UI updates
                self._perform_ik_with_ui_updates(robot_name, target_position, target_orientation)
                
        except Exception as e:
            self.__console.error(f"Failed to perform TCP move: {e}")

    def on_tcp_movestop(self):
        """Stop ongoing IK computation"""
        self.__ik_stop_flag = True
        self.__console.info("TCP move operation stop requested")

    def on_set_initial_pose(self):
        """Set all robot joints to initial pose (0 position)"""
        try:
            # Reset stop flag when starting new operation
            self.__ik_stop_flag = False
            
            self.__console.info("Setting all robot joints to initial pose (0 position)")
            
            # Get all available robots
            robot_names = self.__kinematics.get_robot_names()
            
            for robot_name in robot_names:
                # Get all joint names for this robot
                joint_names = self.__kinematics.get_joint_names(robot_name)
                
                # Create zero joint angles dictionary
                zero_joint_angles = {}
                for joint_name in joint_names:
                    zero_joint_angles[joint_name] = 0.0
                
                self.__console.debug(f"Setting {robot_name} joints to zero: {list(zero_joint_angles.keys())}")
                
                # Set joint angles to zero in manipulation module
                if self.__kinematics.set_joint_angles(robot_name, zero_joint_angles):
                    # Compute forward kinematics to get new end-effector pose
                    fk_result = self.__kinematics.compute_fk(robot_name)
                    
                    if fk_result:
                        # Update TCP view table with new position
                        self.__tcpview_model.add_robot_tcp(
                            robot_name, 
                            fk_result['position'], 
                            fk_result['orientation']
                        )
                        
                        self.__console.info(f"Reset {robot_name}: TCP at {fk_result['position']}")
                        
                        # Send joint angles to viewer3d for visualization
                        for joint_name in joint_names:
                            self.__call(socket=self.__socket, function="API_set_joint_angle", 
                                       kwargs={"joint": joint_name, "value": 0.0})
                    else:
                        self.__console.error(f"Failed to compute FK for {robot_name} after reset")
                else:
                    self.__console.error(f"Failed to set joint angles for {robot_name}")
            
            # Update all UI controls to reflect zero positions
            self._update_all_joint_ui_controls()
            
            self.__console.info("Successfully set all robot joints to initial pose")
                
        except Exception as e:
            self.__console.error(f"Failed to set initial pose: {e}")

    def on_slide_control_update(self, value, joint:str):
        slider = self.sender()
        self.findChild(QLineEdit, f"edit_{slider.objectName()}").setText(str(value/JOINT_ANGLE_SCALE_FACTOR))
        
        # Convert to radians
        angle_rad = value/JOINT_ANGLE_SCALE_FACTOR*3.14/180
        
        # Send to viewer3d for visualization
        self.__call(socket=self.__socket, function="API_set_joint_angle", kwargs={"joint": joint, "value": angle_rad})
        
        # Update manipulation module and compute FK
        # Determine robot name from joint name (joint format: "dda_joint_xxx" or "rt_joint_xxx")
        robot_name = None
        
        # Extract prefix from joint name (e.g., "dda" from "dda_joint_base")
        if "_joint_" in joint:
            joint_prefix = joint.split("_joint_")[0]  # "dda" or "rt"
            
            # Find matching robot name in config
            for robot in self.__kinematics.get_robot_names():
                if joint_prefix in robot:  # "dda" matches "dda_rb10_1300e"
                    robot_name = robot
                    break
        
        self.__console.debug(f"Joint {joint} mapped to robot {robot_name}")
        
        if robot_name:
            self.update_robot_joint_angle(robot_name, joint, angle_rad)
        else:
            self.__console.warning(f"Could not find robot for joint {joint}")
        
        self.__console.info(f"({self.__class__.__name__}) Joint {joint} value changed to {angle_rad:.4f} rad ({np.rad2deg(angle_rad):.1f}°)")

    def on_geometry_transform_changed(self, name: str, position: list, orientation: list):
        """Handle geometry transform changes from table"""
        self.__console.info(f"Geometry {name} transform changed: pos={position}, ori={orientation}")
        self.__call(socket=self.__socket, function="API_update_geometry_transform", kwargs={"name": name, "pos": position, "ori": orientation})

    def on_geometry_table_key_press(self, event):
        """Handle key press events for geometry table"""
        from PyQt6.QtCore import Qt
        
        if event.key() == Qt.Key.Key_Delete:
            # Get selected indexes
            selected_indexes = self.table_geometry.selectionModel().selectedIndexes()
            if selected_indexes:
                # Delete from model and get deleted names
                deleted_names = self.__geometry_model.delete_selected_geometry(selected_indexes)
                for name in deleted_names:
                    self.__call(socket=self.__socket, function="API_remove_geometry", kwargs={"name": name})
                
                self.__console.info(f"Deleted {len(deleted_names)} geometry objects")
        else:
            # Call the original keyPressEvent for other keys
            super(type(self.table_geometry), self.table_geometry).keyPressEvent(event)

    def _setup_sliders(self, dda_side_joint_limits:dict, rt_side_joint_limits:dict):
        """Setup sliders with URDF joint limits"""
        try:
            # RT-side Manipulator joint control
            rt_labels = self.findChildren(QLabel, QRegularExpression(r"label_rt_\d+"))
            rt_edits = self.findChildren(QLineEdit, QRegularExpression(r"edit_slide_rt_\d+"))
            rt_sliders = self.findChildren(QSlider, QRegularExpression(r"slide_rt_\d+"))
            rt_labels.sort(key=lambda lbl: int(re.search(r"\d+", lbl.objectName()).group()))
            rt_edits.sort(key=lambda lbl: int(re.search(r"\d+", lbl.objectName()).group()))
            rt_sliders.sort(key=lambda lbl: int(re.search(r"\d+", lbl.objectName()).group()))

            for label, key in zip(rt_labels, rt_side_joint_limits.keys()):
                label.setText(key)
            for label, slider, edit, key in zip(rt_labels, rt_sliders, rt_edits, rt_side_joint_limits.keys()):
                # slider.valueChanged.connect(lambda v: self.on_slide_control_update(v, key))
                slider.valueChanged.connect(partial(self.on_slide_control_update, joint=key))
                
                # Map joint name to UI controls
                self.__joint_ui_mapping[key] = {
                    'slider': slider,
                    'edit': edit
                }

                limits = rt_side_joint_limits[label.text()]
                lower_deg = int(limits['lower'] * 180 / 3.14)*JOINT_ANGLE_SCALE_FACTOR
                upper_deg = int(limits['upper'] * 180 / 3.14)*JOINT_ANGLE_SCALE_FACTOR
                slider.setRange(lower_deg, upper_deg)
                slider.setValue(0)
                edit.setText(str(0))
        except Exception as e:
            self.__console.warning(f"{e}")

        try:
            # DDA-side Manipulator joint control
            dda_labels = self.findChildren(QLabel, QRegularExpression(r"label_dda_\d+"))
            dda_edits = self.findChildren(QLineEdit, QRegularExpression(r"edit_slide_dda_\d+"))
            dda_sliders = self.findChildren(QSlider, QRegularExpression(r"slide_dda_\d+"))
            dda_labels.sort(key=lambda lbl: int(re.search(r"\d+", lbl.objectName()).group()))
            dda_edits.sort(key=lambda lbl: int(re.search(r"\d+", lbl.objectName()).group()))
            dda_sliders.sort(key=lambda lbl: int(re.search(r"\d+", lbl.objectName()).group()))

            for label, key in zip(dda_labels, dda_side_joint_limits.keys()):
                label.setText(key)
            for label, slider, edit, key in zip(dda_labels, dda_sliders, dda_edits, dda_side_joint_limits.keys()):
                # slider.valueChanged.connect(lambda v: self.on_slide_control_update(v, key))
                slider.valueChanged.connect(partial(self.on_slide_control_update, joint=key))
                
                # Map joint name to UI controls
                self.__joint_ui_mapping[key] = {
                    'slider': slider,
                    'edit': edit
                }

                limits = dda_side_joint_limits[label.text()]
                lower_deg = int(limits['lower'] * 180 / 3.14)*JOINT_ANGLE_SCALE_FACTOR
                upper_deg = int(limits['upper'] * 180 / 3.14)*JOINT_ANGLE_SCALE_FACTOR
                slider.setRange(lower_deg, upper_deg)
                slider.setValue(0)
                edit.setText(str(0))
        except Exception as e:
            self.__console.warning(f"{e}")

        

    def on_select_generate_pcd(self):
        """ 3D Model to PCD file generation """
        from python.manager.util_gen_pcd import AppWindow as GenPCDWindow
        model_file, _ = QFileDialog.getOpenFileName(self, "Open STL File", "", "STL Files (*.stl);;All Files (*)")
        if model_file:
            wnd = GenPCDWindow(config=self.__config, target=model_file)
            wnd.show()

            self.__console.info(f"Selected 3D model file: {model_file}")
        
    def __call(self, socket, function:str, kwargs:dict):
        """ call function via zpipe to others"""
        try:
            topic = "call"
            message = { "function":function,"kwargs":kwargs}
            if socket:
                socket.dispatch([topic, json.dumps(message)])
            else:
                self.__console.warning(f"Failed send")
            
        except zmq.ZMQError as e:
            self.__console.error(f"[Main Window] {e}")
        except json.JSONDecodeError as e:
            self.__console.error(f"[Main Window] JSON Decode Error: {e}")


    # ========== Manipulation Module Methods ==========
    
    def update_robot_joint_angle(self, robot_name: str, joint_name: str, angle_rad: float):
        """Update joint angle and compute forward kinematics"""
        try:
            self.__console.debug(f"Updating joint {joint_name} on robot {robot_name} to {angle_rad:.4f} rad")
            
            # Update joint angle in manipulation module
            if self.__kinematics.set_joint_angle(robot_name, joint_name, angle_rad):
                self.__console.debug(f"Joint angle set successfully, computing FK...")
                
                # Compute forward kinematics to get end-effector pose
                fk_result = self.__kinematics.compute_fk(robot_name)
                
                if fk_result:
                    # self.__console.debug(f"FK result: pos={fk_result['position']}, ori={fk_result['orientation']}")
                    
                    # Update TCP view table with new end-effector position
                    self.__tcpview_model.add_robot_tcp(
                        robot_name, 
                        fk_result['position'], 
                        fk_result['orientation']
                    )
                    
                    self.__console.info(f"Updated {robot_name}.{joint_name} = {np.rad2deg(angle_rad):.1f}°, "f"TCP: {fk_result['position']}")
                    return True
                else:
                    self.__console.error(f"FK computation failed for {robot_name}")
            else:
                self.__console.error(f"Failed to set joint angle for {robot_name}.{joint_name}")
                    
        except Exception as e:
            self.__console.error(f"Failed to update robot joint angle: {e}")
        return False
    
    def update_robot_joint_angles(self, robot_name: str, joint_angles: dict):
        """Update multiple joint angles and compute forward kinematics"""
        try:
            # Update joint angles in manipulation module
            if self.__kinematics.set_joint_angles(robot_name, joint_angles):
                # Compute forward kinematics to get end-effector pose
                fk_result = self.__kinematics.compute_fk(robot_name)
                
                if fk_result:
                    # Update TCP view table with new end-effector position
                    self.__tcpview_model.add_robot_tcp(
                        robot_name, 
                        fk_result['position'], 
                        fk_result['orientation']
                    )
                    
                    self.__console.debug(f"Updated {robot_name} joints, TCP: {fk_result['position']}")
                    return True
                    
        except Exception as e:
            self.__console.error(f"Failed to update robot joint angles: {e}")
        return False
    
    def get_robot_end_effector_pose(self, robot_name: str):
        """Get current end-effector pose for a robot"""
        try:
            return self.__kinematics.get_end_effector_pose(robot_name)
        except Exception as e:
            self.__console.error(f"Failed to get end-effector pose for {robot_name}: {e}")
            return None
    
    def get_all_robot_poses(self):
        """Get end-effector poses for all robots"""
        try:
            return self.__kinematics.compute_all_robots_fk()
        except Exception as e:
            self.__console.error(f"Failed to get all robot poses: {e}")
            return {}
    
    def reset_robot_joints(self, robot_name: str = None):
        """Reset robot joints to zero position"""
        try:
            self.__kinematics.reset_joint_angles(robot_name)
            
            # Update TCP view table
            if robot_name:
                fk_result = self.__kinematics.compute_fk(robot_name)
                if fk_result:
                    self.__tcpview_model.add_robot_tcp(
                        robot_name, 
                        fk_result['position'], 
                        fk_result['orientation']
                    )
            else:
                # Reset all robots
                for name in self.__kinematics.get_robot_names():
                    fk_result = self.__kinematics.compute_fk(name)
                    if fk_result:
                        self.__tcpview_model.add_robot_tcp(
                            name, 
                            fk_result['position'], 
                            fk_result['orientation']
                        )
            
            self.__console.info(f"Reset robot joints: {robot_name if robot_name else 'all robots'}")
            return True
            
        except Exception as e:
            self.__console.error(f"Failed to reset robot joints: {e}")
            return False
    
    def get_robot_info(self, robot_name: str):
        """Get robot information including joint names and current configuration"""
        try:
            return self.__kinematics.get_robot_info(robot_name)
        except Exception as e:
            self.__console.error(f"Failed to get robot info for {robot_name}: {e}")
            return None

    def compute_ik(self, robot_name: str, target_position: list, target_orientation: list = None):
        """Compute inverse kinematics and update robot joints"""
        try:
            ik_result = self.__kinematics.compute_ik(
                robot_name, target_position, target_orientation, max_iterations=1000, tolerance=1
            )
            
            if ik_result and ik_result['success']:
                # Update manipulation module with IK result
                if self.__kinematics.set_joint_angles(robot_name, ik_result['joint_angles']):
                    # Compute forward kinematics to verify and update TCP table
                    fk_result = self.__kinematics.compute_fk(robot_name)
                    
                    if fk_result:
                        # Update TCP view table with new position
                        self.__tcpview_model.add_robot_tcp(
                            robot_name, 
                            fk_result['position'], 
                            fk_result['orientation']
                        )
                        
                        self.__console.info(f"IK success for {robot_name}: {ik_result['iterations']} iterations, "f"error={ik_result['error']:.6f}")
                        return ik_result
                        
            else:
                self.__console.warning(f"IK failed for {robot_name}")
                return None
                
        except Exception as e:
            self.__console.error(f"Failed to compute inverse kinematics: {e}")
            return None

    def set_target_tcp_pose(self, robot_name: str, position: list, orientation: list = None):
        """Set target TCP pose and solve inverse kinematics"""
        return self.compute_ik(robot_name, position, orientation)

    def _initialize_tcp_table(self):
        """Initialize TCP table with initial robot end-effector poses (all joints at 0)"""
        try:
            self.__console.info("Initializing TCP table with robot end-effector poses...")
            
            # Get all available robots
            robot_names = self.__kinematics.get_robot_names()
            
            for robot_name in robot_names:
                # Compute forward kinematics with all joints at 0 (default initialization)
                fk_result = self.__kinematics.compute_fk(robot_name)
                
                if fk_result:
                    # Add robot TCP to the table
                    self.__tcpview_model.add_robot_tcp(robot_name, fk_result['position'], fk_result['orientation'])
                    # self.__console.debug(f"Initialized TCP for {robot_name}: "f"pos={fk_result['position']}, " f"ori_deg={fk_result['orientation_deg']}")
                else:
                    self.__console.error(f"Failed to compute initial FK for robot {robot_name}")
            
            self.__console.info(f"TCP table initialized with {len(robot_names)} robots")
            
        except Exception as e:
            self.__console.error(f"Failed to initialize TCP table: {e}")
    
    def _update_joint_ui_controls(self, joint_angles: dict):
        """Update slider and line edit controls with new joint angles"""
        try:
            self.__console.info(f"_update_joint_ui_controls called with {len(joint_angles)} joints")
            self.__console.info(f"Available UI mappings: {list(self.__joint_ui_mapping.keys())}")
            
            for joint_name, angle_rad in joint_angles.items():
                self.__console.info(f"Processing joint {joint_name}: {angle_rad:.4f} rad")
                
                if joint_name in self.__joint_ui_mapping:
                    controls = self.__joint_ui_mapping[joint_name]
                    slider = controls['slider']
                    edit = controls['edit']
                    
                    # Convert radians to degrees and scale for slider
                    angle_deg = angle_rad * 180.0 / math.pi
                    slider_value = int(angle_deg * JOINT_ANGLE_SCALE_FACTOR)
                    
                    self.__console.info(f"Converting {joint_name}: {angle_rad:.4f} rad -> {angle_deg:.2f}° -> slider:{slider_value}")
                    
                    # Get current values for comparison
                    current_slider_value = slider.value()
                    current_edit_text = edit.text()
                    
                    # Temporarily disconnect signals to avoid recursive updates
                    slider.blockSignals(True)
                    edit.blockSignals(True)
                    
                    # Update slider and line edit
                    slider.setValue(slider_value)
                    edit.setText(f"{angle_deg:.2f}")
                    
                    # Re-enable signals
                    slider.blockSignals(False)
                    edit.blockSignals(False)
                    
                    self.__console.info(f"Updated {joint_name}: slider {current_slider_value}->{slider.value()}, edit '{current_edit_text}'->'{edit.text()}'")
                else:
                    self.__console.warning(f"No UI mapping found for joint {joint_name}")
                    
        except Exception as e:
            self.__console.error(f"Failed to update joint UI controls: {e}")
    
    def _update_all_joint_ui_controls(self):
        """Update all joint UI controls to reflect current joint angles in manipulation module"""
        try:
            robot_names = self.__kinematics.get_robot_names()
            
            for robot_name in robot_names:
                # Get current joint angles from manipulation module
                current_joint_angles = self.__kinematics.get_joint_angles(robot_name)
                
                if current_joint_angles:
                    self.__console.debug(f"Updating UI controls for {robot_name} with {len(current_joint_angles)} joints")
                    self._update_joint_ui_controls(current_joint_angles)
                else:
                    self.__console.warning(f"No joint angles found for {robot_name}")
                    
        except Exception as e:
            self.__console.error(f"Failed to update all joint UI controls: {e}")
    
    def _perform_ik_with_ui_updates(self, robot_name: str, target_position: list, target_orientation: list):
        """Perform IK with UI updates at each iteration"""
        try:
            # Create callback function for iteration updates
            def iteration_callback(iteration: int, joint_angles: dict, error: float):
                """Called at each IK iteration to update UI"""
                # Check if stop flag is set
                if self.__ik_stop_flag:
                    self.__console.info("IK iteration stopped by user request")
                    return False  # Signal to stop iteration
                
                self.__console.info(f"IK Iteration {iteration}: error={error:.6f}")
                
                # Update UI controls with current joint angles
                self._update_joint_ui_controls(joint_angles)
                
                # Send updated joint angles to viewer3d for visualization
                for joint_name, angle in joint_angles.items():
                    self.__call(socket=self.__socket, function="API_set_joint_angle", kwargs={"joint": joint_name, "value": angle})
                
                # Allow GUI to update
                QApplication.processEvents()
                
                # Small delay to visualize the iteration process
                import time
                time.sleep(0.1)
                
                return True  # Continue iteration
            
            # Perform inverse kinematics with iteration callback
            ik_result = self.__kinematics.compute_ik_with_callback( robot_name, target_position, target_orientation, 
                iteration_callback=iteration_callback, max_iterations=100, tolerance=1e-3)
            
            if ik_result and ik_result['success']:
                self.__console.info(f"IK converged for {robot_name} in {ik_result['iterations']} iterations, error={ik_result['error']:.6f}")
            else:
                if self.__ik_stop_flag:
                    self.__console.info(f"IK stopped for {robot_name} by user request")
                else:
                    self.__console.warning(f"IK failed to converge for {robot_name}")
                
        except Exception as e:
            self.__console.error(f"Failed to perform IK with UI updates: {e}")


