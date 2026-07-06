# Overview
입력된 ply 파일과 임의 3차원 좌표를 주었을때 해당 위치와 가장 가까운 포인트를 기준으로 실린더타입의 primitive shape를 detection하는 알고리즘을 만들고, 그 결과를 visualization하는 프로그램

# Input Arguments
- Point Cloud Data File (*.ply) path
- Input Point(x,y,z)
- cylinder height : 0.4

# Output
- Cylinder Center Axis : origin [px, py, pz], direction [nx, ny, nz]
- cylinder radius : R

# Dependencies
- Open3D or Vedo Library

# Method
- RANSAC

# Usage
- python 3d_cylinder_shape_detection.py --pcd test_pipe.ply --in_csv test_pipe.csv --height 0.1