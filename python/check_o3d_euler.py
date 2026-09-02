import open3d as o3d
import numpy as np

# Test 1: ZYX vs XYZ
# angles = [0.1, 0.2, 0.3]
# R_o3d = o3d.geometry.get_rotation_matrix_from_xyz(angles)

# Construct manual matrices
def Rz(t): return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0,0,1]])
def Ry(t): return np.array([[np.cos(t), 0, np.sin(t)], [0,1,0], [-np.sin(t), 0, np.cos(t)]])
def Rx(t): return np.array([[1,0,0], [0,np.cos(t), -np.sin(t)], [0,np.sin(t), np.cos(t)]])

angles = np.array([0.5, 0.3, 0.8])
R_o3d = o3d.geometry.get_rotation_matrix_from_xyz(angles)

R_xyz = Rx(angles[0]) @ Ry(angles[1]) @ Rz(angles[2])
R_zyx = Rz(angles[2]) @ Ry(angles[1]) @ Rx(angles[0])

print(f"O3D:\n{R_o3d}")
print(f"XYZ:\n{R_xyz}")
print(f"ZYX:\n{R_zyx}")

if np.allclose(R_o3d, R_xyz):
    print("MATCHES SCENARIO: R = Rx * Ry * Rz (XYZ)")
elif np.allclose(R_o3d, R_zyx):
    print("MATCHES SCENARIO: R = Rz * Ry * Rx (ZYX)")
else:
    print("NO MATCH")
