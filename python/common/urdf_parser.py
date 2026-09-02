"""
URDF Parser for extracting joint limits
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
"""

import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional
import os

class URDFParser:
    def __init__(self, urdf_path: str):
        """
        Initialize URDF parser with path to URDF file
        
        Args:
            urdf_path (str): Path to the URDF file
        """
        self.urdf_path = urdf_path
        self.joint_limits = {}
        self._parse_urdf()
    
    def _parse_urdf(self):
        """Parse URDF file and extract joint limits"""
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            
            # Find all joints with limits
            for joint in root.findall('joint'):
                joint_name = joint.get('name')
                joint_type = joint.get('type')
                
                # Only process revolute and prismatic joints
                if joint_type in ['revolute', 'prismatic']:
                    limit_elem = joint.find('limit')
                    if limit_elem is not None:
                        lower = float(limit_elem.get('lower', '-3.14'))
                        upper = float(limit_elem.get('upper', '3.14'))
                        effort = float(limit_elem.get('effort', '10'))
                        velocity = float(limit_elem.get('velocity', '3.14'))
                        
                        self.joint_limits[joint_name] = {
                            'lower': lower,
                            'upper': upper,
                            'effort': effort,
                            'velocity': velocity,
                            'type': joint_type
                        }
        
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse URDF file: {e}")
    
    def get_joint_limits(self, joint_name: str) -> Optional[Dict]:
        """
        Get joint limits for a specific joint
        
        Args:
            joint_name (str): Name of the joint
            
        Returns:
            Dict: Joint limits dictionary or None if joint not found
        """
        return self.joint_limits.get(joint_name)
    
    def get_joint_range(self, joint_name: str) -> Optional[Tuple[float, float]]:
        """
        Get joint range (lower, upper) for a specific joint
        
        Args:
            joint_name (str): Name of the joint
            
        Returns:
            Tuple[float, float]: (lower, upper) limits or None if joint not found
        """
        limits = self.get_joint_limits(joint_name)
        if limits:
            return (limits['lower'], limits['upper'])
        return None
    
    def get_all_joint_limits(self) -> Dict:
        """
        Get all joint limits
        
        Returns:
            Dict: Dictionary of all joint limits
        """
        return self.joint_limits.copy()
    
    def get_joint_names(self) -> list:
        """
        Get list of all joint names with limits
        
        Returns:
            list: List of joint names
        """
        return list(self.joint_limits.keys())
