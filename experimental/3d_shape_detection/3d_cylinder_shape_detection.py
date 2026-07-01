import open3d as o3d
import numpy as np
import argparse
import sys
import os
import pandas as pd

def create_test_cylinder(file_path):
    """Generate a synthetic cylinder for testing if no file is provided."""
    print(f"Creating synthetic cylinder at {file_path}...")
    radius = 0.1
    height = 0.4
    
    n_points = 5000
    z = np.random.uniform(-height/2, height/2, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    points = np.stack([x, y, z], axis=1)
    points += np.random.normal(0, 0.002, points.shape)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)
    return pcd

def fit_cylinder(pcd, input_point, height_limit=0.4, search_radius=0.2):
    """Detect a cylinder near input_point."""
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(input_point, search_radius)
    if k < 10:
        return None
    
    neighborhood = pcd.select_by_index(idx)
    neighborhood.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    points = np.asarray(neighborhood.points)
    normals = np.asarray(neighborhood.normals)
    
    cov = np.cov(normals.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis_dir = eigenvectors[:, 0]
    
    ref_vec = np.array([1, 0, 0]) if abs(axis_dir[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(axis_dir, ref_vec)
    u /= np.linalg.norm(u)
    v = np.cross(axis_dir, u)
    
    mean_pts = np.mean(points, axis=0)
    pts_centered = points - mean_pts
    pts_2d = np.stack([np.dot(pts_centered, u), np.dot(pts_centered, v)], axis=1)
    
    best_r = 0
    best_center_2d = np.array([0, 0])
    max_inliers = 0
    
    num_iterations = 500
    for _ in range(num_iterations):
        if len(pts_2d) < 3: break
        sample_idx = np.random.choice(len(pts_2d), 3, replace=False)
        p1, p2, p3 = pts_2d[sample_idx]
        
        A = np.array([
            [2*p1[0], 2*p1[1], 1],
            [2*p2[0], 2*p2[1], 1],
            [2*p3[0], 2*p3[1], 1]
        ])
        b = np.array([
            p1[0]**2 + p1[1]**2,
            p2[0]**2 + p2[1]**2,
            p3[0]**2 + p3[1]**2
        ])
        try:
            sol = np.linalg.solve(A, b)
            xc, yc = sol[0], sol[1]
            r = np.sqrt(max(0, sol[2] + xc**2 + yc**2))
            
            dists = np.abs(np.sqrt((pts_2d[:, 0] - xc)**2 + (pts_2d[:, 1] - yc)**2) - r)
            inliers = dists < 0.005
            num_inliers = np.sum(inliers)
            
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_r = r
                best_center_2d = np.array([xc, yc])
        except np.linalg.LinAlgError:
            continue
            
    if max_inliers < 10:
        return None
    
    axis_origin = mean_pts + best_center_2d[0]*u + best_center_2d[1]*v
    
    return {
        "origin": axis_origin,
        "direction": axis_dir,
        "radius": best_r
    }

def main():
    parser = argparse.ArgumentParser(description="3D Cylinder Shape Detection")
    parser.add_argument("--pcd", type=str, help="Path to point cloud data file (*.ply)")
    parser.add_argument("--in", dest="input_point_str", type=str, help="Input Point (x,y,z)")
    parser.add_argument("--in_csv", type=str, help="CSV file containing input points (columns: x, y, z)")
    parser.add_argument("--height", type=float, default=0.4, help="Cylinder height (default: 0.4)")
    
    args = parser.parse_args()
    
    if not args.pcd:
        pcd_path = "test_pipe.ply"
        if not os.path.exists(pcd_path):
            pcd = create_test_cylinder(pcd_path)
        else:
            pcd = o3d.io.read_point_cloud(pcd_path)
    else:
        if not os.path.exists(args.pcd):
            print(f"File not found: {args.pcd}")
            sys.exit(1)
        pcd = o3d.io.read_point_cloud(args.pcd)
        
    input_points = []
    if args.in_csv:
        if not os.path.exists(args.in_csv):
            print(f"CSV file not found: {args.in_csv}")
            sys.exit(1)
        df = pd.read_csv(args.in_csv)
        if not all(col in df.columns for col in ['x', 'y', 'z']):
            print("CSV must have 'x', 'y', 'z' columns.")
            sys.exit(1)
        input_points = df[['x', 'y', 'z']].values
    elif args.input_point_str:
        try:
            input_points = [np.array([float(x) for x in args.input_point_str.split(',')])]
        except ValueError:
            print("Invalid input point format. Use x,y,z")
            sys.exit(1)
    else:
        input_points = [np.asarray(pcd.get_center())]
        print(f"No input points provided. Using cloud center: {input_points[0]}")

    print(f"\n{'='*60}")
    print(f"Processing {len(input_points)} input points...")
    print(f"{'='*60}")
    
    vis_geometries = [pcd]
    results_found = 0
    
    for i, pt in enumerate(input_points):
        print(f"\n[Point {i+1}/{len(input_points)}] Input: {pt}")
        result = fit_cylinder(pcd, pt, height_limit=args.height)
        
        if result:
            results_found += 1
            print(f"  > Detection Success!")
            print(f"    - Origin: {result['origin']}")
            print(f"    - Direction: {result['direction']}")
            print(f"    - Radius: {result['radius']:.6f}")
            
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=result['radius'], height=args.height)
            color = [0.1, 0.9, 0.1] if len(input_points) == 1 else list(np.random.rand(3))
            mesh_cylinder.paint_uniform_color(color)
            
            target_dir = result['direction']
            target_dir /= np.linalg.norm(target_dir)
            
            z_axis = np.array([0, 0, 1])
            v_cross = np.cross(z_axis, target_dir)
            c = np.dot(z_axis, target_dir)
            s = np.linalg.norm(v_cross)
            
            if s > 1e-6:
                kmat = np.array([[0, -v_cross[2], v_cross[1]], [v_cross[2], 0, -v_cross[0]], [-v_cross[1], v_cross[0], 0]])
                rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            else:
                rotation_matrix = np.eye(3) if c > 0 else -np.eye(3)
                
            mesh_cylinder.rotate(rotation_matrix, center=(0, 0, 0))
            mesh_cylinder.translate(result['origin'])
            
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=result['origin'])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(pt)
            sphere.paint_uniform_color([1, 0, 0])
            
            vis_geometries.extend([mesh_cylinder, coord, sphere])
        else:
            print("  > Detection failed for this point.")

    print(f"\n{'='*60}")
    print(f"Total detections: {results_found}/{len(input_points)}")
    print(f"{'='*60}")
    
    if results_found > 0:
        print("\nVisualizing all results... Close the window to exit.")
        o3d.visualization.draw_geometries(vis_geometries)
    else:
        print("\nNo cylinders were detected.")

if __name__ == "__main__":
    main()
