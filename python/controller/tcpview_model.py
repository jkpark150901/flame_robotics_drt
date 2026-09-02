"""
TCP View Table Model for QTableView
Manages TCP (Tool Center Point) positions and orientations for robot arms
@author: Byunghun Hwang (bh.hwang@iae.re.kr)
"""

from PyQt6.QtCore import QAbstractTableModel, Qt, pyqtSignal
from PyQt6.QtGui import QColor
import numpy as np
from util.logger.console import ConsoleLogger


class TCPViewTableModel(QAbstractTableModel):
    """Table model for TCP positions and orientations in global coordinate system"""
    
    # Signal emitted when TCP transform is updated
    tcpTransformChanged = pyqtSignal(str, list, list)  # robot_name, position, orientation
    
    def __init__(self, config:dict):
        super().__init__()
        self.__console = ConsoleLogger.get_logger()
        self.__config = config
        
        # Column definitions for TCP data
        self.columns = [
            "Robot Name", "X", "Y", "Z", "X-Axis Rot (rad)", "Y-Axis Rot (rad)", "Z-Axis Rot (rad)"
        ]
        
        # Data storage: dict with robot name as key
        # Value: {"pos": [x, y, z], "ori": [rx, ry, rz]} in global coordinate system
        self.tcp_data = {}
        
        # Initialize with robot names from config
        self._initialize_robots_from_config()
        
    def _initialize_robots_from_config(self):
        """Initialize table with robot names from config URDF section"""
        if "urdf" in self.__config:
            for urdf_config in self.__config["urdf"]:
                robot_name = urdf_config.get("name", "unknown")
                self.tcp_data[robot_name] = {
                    'pos': [0.0, 0.0, 0.0],  # Default TCP position (will be updated by FK)
                    'ori': [0.0, 0.0, 0.0]   # Default TCP orientation in radians (will be updated by FK)
                }
                self.__console.debug(f"Initialized TCP data for robot: {robot_name}")
        else:
            self.__console.warning("No URDF configuration found in config for TCP view initialization")
    
    def rowCount(self, parent=None):
        return len(self.tcp_data)
    
    def columnCount(self, parent=None):
        return len(self.columns)
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section]
        return None
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
            
        row = index.row()
        col = index.column()
        
        if row >= len(self.tcp_data):
            return None
            
        robot_names = list(self.tcp_data.keys())
        robot_name = robot_names[row]
        tcp_info = self.tcp_data[robot_name]
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if col == 0:  # Robot Name
                return robot_name
            elif col in [1, 2, 3]:  # X, Y, Z position
                pos_idx = col - 1
                return f"{tcp_info['pos'][pos_idx]:.3f}"
            elif col in [4, 5, 6]:  # X, Y, Z axis rotation (display in radians)
                rot_idx = col - 4
                radians = tcp_info['ori'][rot_idx]
                return f"{radians:.4f}"
        
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Different colors for different robot types
            if "ur" in robot_name.lower():
                return QColor(255, 240, 245)  # Light pink for UR robots
            elif "rb" in robot_name.lower():
                return QColor(240, 255, 240)  # Light green for RB robots
            elif "rt" in robot_name.lower():
                return QColor(240, 248, 255)  # Light blue for RT robots
            elif "dda" in robot_name.lower():
                return QColor(255, 248, 220)  # Light yellow for DDA robots
            else:
                return QColor(248, 248, 248)  # Light gray for others
                
        return None
    
    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
            
        row = index.row()
        col = index.column()
        
        if row >= len(self.tcp_data) or col == 0:  # Can't edit robot name
            return False
            
        try:
            float_value = float(value)
        except ValueError:
            self.__console.warning(f"Invalid numeric value: {value}")
            return False
            
        robot_names = list(self.tcp_data.keys())
        robot_name = robot_names[row]
        tcp_info = self.tcp_data[robot_name]
        
        # Update the data
        if col in [1, 2, 3]:  # Position (X, Y, Z)
            pos_idx = col - 1
            tcp_info['pos'][pos_idx] = float_value
        elif col in [4, 5, 6]:  # Orientation (input is already in radians)
            rot_idx = col - 4
            tcp_info['ori'][rot_idx] = float_value
        
        # Emit signal for TCP update
        self.tcpTransformChanged.emit(
            robot_name, 
            tcp_info['pos'].copy(), 
            tcp_info['ori'].copy()
        )
        
        self.dataChanged.emit(index, index)
        return True
    
    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
            
        # Robot name column is not editable, others are
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        else:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
    
    def add_robot_tcp(self, robot_name: str, pos: list = [0, 0, 0], ori: list = [0, 0, 0]):
        """Add or update robot TCP data"""
        if robot_name in self.tcp_data:
            # Update existing robot TCP
            self.tcp_data[robot_name]['pos'] = pos.copy()
            self.tcp_data[robot_name]['ori'] = ori.copy()
            self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount()-1, self.columnCount()-1))
        else:
            # Add new robot TCP
            self.beginInsertRows(self.index(-1, -1), self.rowCount(), self.rowCount())
            self.tcp_data[robot_name] = {
                'pos': pos.copy(),
                'ori': ori.copy()
            }
            self.endInsertRows()
        
        self.__console.debug(f"Added robot TCP to table: {robot_name} at {pos} with rotation {ori}")
    
    def remove_robot_tcp(self, robot_name: str):
        """Remove robot TCP from the table"""
        if robot_name not in self.tcp_data:
            return
            
        robot_names = list(self.tcp_data.keys())
        row = robot_names.index(robot_name)
        
        self.beginRemoveRows(self.index(-1, -1), row, row)
        del self.tcp_data[robot_name]
        self.endRemoveRows()
        
        self.__console.debug(f"Removed robot TCP from table: {robot_name}")
    
    def clear_all_tcp(self):
        """Clear all TCP data from the table"""
        self.beginResetModel()
        self.tcp_data.clear()
        self._initialize_robots_from_config()  # Re-initialize with config robots
        self.endResetModel()
        
        self.__console.debug("Cleared all TCP data from table")
    
    def get_tcp_info(self, robot_name: str):
        """Get TCP information by robot name"""
        return self.tcp_data.get(robot_name, None)
    
    def update_tcp_position(self, robot_name: str, pos: list):
        """Update only the position of a robot's TCP"""
        if robot_name in self.tcp_data:
            self.tcp_data[robot_name]['pos'] = pos.copy()
            
            # Find row and emit data changed signal
            robot_names = list(self.tcp_data.keys())
            if robot_name in robot_names:
                row = robot_names.index(robot_name)
                start_index = self.createIndex(row, 1)  # X column
                end_index = self.createIndex(row, 3)    # Z column
                self.dataChanged.emit(start_index, end_index)
    
    def update_tcp_orientation(self, robot_name: str, ori: list):
        """Update only the orientation of a robot's TCP"""
        if robot_name in self.tcp_data:
            self.tcp_data[robot_name]['ori'] = ori.copy()
            
            # Find row and emit data changed signal
            robot_names = list(self.tcp_data.keys())
            if robot_name in robot_names:
                row = robot_names.index(robot_name)
                start_index = self.createIndex(row, 4)  # X-Axis Rot column
                end_index = self.createIndex(row, 6)    # Z-Axis Rot column
                self.dataChanged.emit(start_index, end_index)
    
    def get_all_robot_names(self):
        """Get list of all robot names in the table"""
        return list(self.tcp_data.keys())
    
    def get_all_tcp_data(self):
        """Get all TCP data for all robots"""
        return self.tcp_data.copy()

    def reset_tcp_to_origin(self, robot_name: str = None):
        """Reset TCP to origin (0,0,0) position and orientation"""
        if robot_name:
            # Reset specific robot
            if robot_name in self.tcp_data:
                self.tcp_data[robot_name]['pos'] = [0.0, 0.0, 0.0]
                self.tcp_data[robot_name]['ori'] = [0.0, 0.0, 0.0]
                
                robot_names = list(self.tcp_data.keys())
                row = robot_names.index(robot_name)
                start_index = self.createIndex(row, 1)
                end_index = self.createIndex(row, 6)
                self.dataChanged.emit(start_index, end_index)
        else:
            # Reset all robots
            for name in self.tcp_data:
                self.tcp_data[name]['pos'] = [0.0, 0.0, 0.0]
                self.tcp_data[name]['ori'] = [0.0, 0.0, 0.0]
            
            self.dataChanged.emit(self.createIndex(0, 1), self.createIndex(self.rowCount()-1, self.columnCount()-1))
        
        self.__console.debug(f"Reset TCP to origin for robot: {robot_name if robot_name else 'all robots'}")