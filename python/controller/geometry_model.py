"""
Geometry Table Model for QTableView
Manages geometry objects with position and orientation data
@author: Byunghun Hwang (bh.hwang@iae.re.kr)
"""

from PyQt6.QtCore import QAbstractTableModel, Qt, pyqtSignal
from PyQt6.QtGui import QColor
import numpy as np
from util.logger.console import ConsoleLogger


class GeometryTableModel(QAbstractTableModel):
    """Table model for geometry objects with position and orientation"""
    
    # Signal emitted when geometry transform is updated
    geometryTransformChanged = pyqtSignal(str, list, list)  # name, position, orientation
    
    def __init__(self):
        super().__init__()
        self.__console = ConsoleLogger.get_logger()
        
        # Column definitions
        self.columns = [
            "Name", "X", "Y", "Z", "Rot X (°)", "Rot Y (°)", "Rot Z (°)"
        ]
        
        # Data storage: dict with geometry name as key
        # Value: {"pos": [x, y, z], "ori": [rx, ry, rz], "type": "pcd/coord_frame/etc"}
        self.geometry_data = {}
        
    def rowCount(self, parent=None):
        return len(self.geometry_data)
    
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
        
        if row >= len(self.geometry_data):
            return None
            
        geometry_names = list(self.geometry_data.keys())
        geometry_name = geometry_names[row]
        geometry_info = self.geometry_data[geometry_name]
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if col == 0:  # Name
                return geometry_name
            elif col in [1, 2, 3]:  # X, Y, Z position
                pos_idx = col - 1
                return f"{geometry_info['pos'][pos_idx]:.3f}"
            elif col in [4, 5, 6]:  # Rot X, Y, Z (convert radians to degrees for display)
                rot_idx = col - 4
                degrees = np.degrees(geometry_info['ori'][rot_idx])
                return f"{degrees:.1f}"
        
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Different colors for different geometry types
            geometry_type = geometry_info.get('type', 'unknown')
            if geometry_type == 'pcd':
                return QColor(240, 248, 255)  # Light blue for point clouds
            elif geometry_type == 'coord_frame':
                return QColor(255, 248, 220)  # Light yellow for coordinate frames
            else:
                return QColor(248, 248, 248)  # Light gray for others
                
        return None
    
    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
            
        row = index.row()
        col = index.column()
        
        if row >= len(self.geometry_data) or col == 0:  # Can't edit name
            return False
            
        try:
            float_value = float(value)
        except ValueError:
            self.__console.warning(f"Invalid numeric value: {value}")
            return False
            
        geometry_names = list(self.geometry_data.keys())
        geometry_name = geometry_names[row]
        geometry_info = self.geometry_data[geometry_name]
        
        # Update the data
        if col in [1, 2, 3]:  # Position
            pos_idx = col - 1
            geometry_info['pos'][pos_idx] = float_value
        elif col in [4, 5, 6]:  # Orientation (convert degrees to radians for storage)
            rot_idx = col - 4
            radians = np.radians(float_value)
            geometry_info['ori'][rot_idx] = radians
        
        # Emit signal for geometry update
        self.geometryTransformChanged.emit(
            geometry_name, 
            geometry_info['pos'].copy(), 
            geometry_info['ori'].copy()
        )
        
        self.dataChanged.emit(index, index)
        return True
    
    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
            
        # Name column is not editable, others are
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        else:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
    
    def add_geometry(self, name: str, pos: list = [0, 0, 0], ori: list = [0, 0, 0], geometry_type: str = "unknown"):
        """Add a new geometry to the table"""
        if name in self.geometry_data:
            # Update existing geometry
            self.geometry_data[name]['pos'] = pos.copy()
            self.geometry_data[name]['ori'] = ori.copy()
            self.geometry_data[name]['type'] = geometry_type
            self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount()-1, self.columnCount()-1))
        else:
            # Add new geometry
            self.beginInsertRows(self.index(-1, -1), self.rowCount(), self.rowCount())
            self.geometry_data[name] = {
                'pos': pos.copy(),
                'ori': ori.copy(),
                'type': geometry_type
            }
            self.endInsertRows()
        
        self.__console.debug(f"Added geometry to table: {name} at {pos} with rotation {ori}")
    
    def remove_geometry(self, name: str):
        """Remove geometry from the table"""
        if name not in self.geometry_data:
            return
            
        geometry_names = list(self.geometry_data.keys())
        row = geometry_names.index(name)
        
        self.beginRemoveRows(self.index(-1, -1), row, row)
        del self.geometry_data[name]
        self.endRemoveRows()
        
        self.__console.debug(f"Removed geometry from table: {name}")
    
    def clear_all_geometry(self):
        """Clear all geometry from the table"""
        self.beginResetModel()
        self.geometry_data.clear()
        self.endResetModel()
        
        self.__console.debug("Cleared all geometry from table")
    
    def get_geometry_info(self, name: str):
        """Get geometry information by name"""
        return self.geometry_data.get(name, None)
    
    def delete_selected_geometry(self, selected_indexes):
        """Delete geometry at selected indexes"""
        if not selected_indexes:
            return []
        
        # Get unique rows from selected indexes
        rows = sorted(set(index.row() for index in selected_indexes), reverse=True)
        deleted_names = []
        
        geometry_names = list(self.geometry_data.keys())
        
        for row in rows:
            if 0 <= row < len(geometry_names):
                geometry_name = geometry_names[row]
                deleted_names.append(geometry_name)
                
                # Remove from model
                self.beginRemoveRows(self.index(-1, -1), row, row)
                del self.geometry_data[geometry_name]
                self.endRemoveRows()
                
                self.__console.debug(f"Deleted geometry from table: {geometry_name}")
        
        return deleted_names
