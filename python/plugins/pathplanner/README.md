# Path Planning Algorithms

이 디렉토리는 3차원 공간에서 RRT 기반의 경로 계획 알고리즘들을 포함하고 있습니다. 모든 알고리즘은 `PlannerBase`를 상속받아 동일한 인터페이스로 구현되어 있으며, 플러그인 형태로 로드하여 사용할 수 있습니다.

## 알고리즘 목록

### 1. RRT (Rapidly-exploring Random Tree)
- **파일**: `rrt.py`
- **설명**: 가장 기본적인 RRT 알고리즘으로, 무작위 샘플링을 통해 트리를 확장하여 경로를 찾습니다.
- **파라미터** (`rrt.json`):
    - `step_size`: 트리 확장 시 한 번에 이동하는 거리.
    - `max_iter`: 최대 반복 횟수.
    - `goal_bias`: 목표 지점을 샘플링할 확률.
    - `workspace_bounds`: 샘플링 영역 (x, y, z).
- **참고 논문**: LaValle, S. M. (1998). Rapidly-exploring random trees: A new tool for path planning.

### 2. RRT-Connect
- **파일**: `rrt_connect.py`
- **설명**: 시작점과 목표 지점에서 동시에 트리를 확장하며, 두 트리가 만날 때까지 반복하는 양방향 RRT 알고리즘입니다. 일반 RRT보다 수렴 속도가 빠릅니다.
- **파라미터** (`rrt_connect.json`):
    - `step_size`, `max_iter`, `workspace_bounds`: RRT와 동일.
- **참고 논문**: Kuffner, J. J., & LaValle, S. M. (2000). RRT-connect: An efficient approach to single-query path planning.

### 3. RRT* (Optimal RRT)
- **파일**: `rrt_star.py`
- **설명**: RRT에 비용(Cost) 개념을 도입하고, 트리가 확장될 때 주변 노드들을 재연결(Rewiring)하여 점진적으로 최적 경로에 수렴하도록 하는 알고리즘입니다.
- **파라미터** (`rrt_star.json`):
    - `search_radius`: 재연결을 위해 검색할 주변 반경.
    - 그 외 파라미터는 RRT와 동일.
- **참고 논문**: Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning.

### 4. Informed RRT*
- **파일**: `informed_rrt_star.py`
- **설명**: 초기 경로가 발견된 후, 비검색 영역에서의 불필요한 샘플링을 줄이기 위해 타원형(Hyperellipsoid) 영역 내에서만 샘플링을 수행하여 최적화 속도를 높인 알고리즘입니다.
- **파라미터** (`informed_rrt_star.json`):
    - `rrt_star.json`과 동일.
- **참고 논문**: Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2014). Informed RRT*: Optimal sampling-based path planning focused via direct sampling of an admissible ellipsoidal heuristic.

### 5. Task-space RRT
- **File**: `algorithms/task_space_rrt.py`
- **Description**: RRT algorithm operating in 6D task space with weighted distance metric.
- **Features**:
  - Sample in 6D space (Position + Orientation) and check collisions for position.
  - Weighted Euclidean distance metric to balance position and orientation importance.
- **Configuration** (`task_space_rrt.json`):
  - `workspace_bounds`: 6D boundaries.
  - `weights`: Dictionary with `pos` and `orient` weights.
- **Reference**: N/A (Standard RRT in configuration space)

### PRM (`prm`)
**Probabilistic RoadMap**. Constructs a visibility graph by sampling N free configurations and connecting them to k-nearest neighbors. Finds a path by querying the roadmap.
- **Config**: `prm.json`
- **Reference**: Kavraki et al., "Probabilistic roadmaps for path planning in high-dimensional configuration spaces" (TRA 1996).

### BIT* (`bit_star`)
**Batch Informed Trees**. Combines the efficiency of graph-based search (like PRM) with the asymptotic optimality of tree-based search (like RRT*). Uses batches of samples constrained by an ellipsoidal heuristic region.
- **Config**: `bit_star.json`
- **Reference**: Gammell et al., "Batch Informed Trees (BIT*): Geometric Planning via Information Theoretic Search" (ICRA 2015).

### 6. Task-space RRT*
- **File**: `algorithms/task_space_rrt_star.py`
- **Description**: Asymptotically optimal RRT* in 6D task space with weighted distance.
- **Features**:
  - Inherits 6D sampling and weighted metric from Task-space RRT.
  - Adds RRT* rewiring and parent selection to optimize path cost.
- **Configuration** (`task_space_rrt_star.json`):
  - `search_radius`: Radius for finding near neighbors for rewiring.
  - `weights`: `pos` and `orient` weights.
- **Reference**: Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning.

## Path Optimization
- **Folder**: `algorithms/optimization/`
- **Usage**: Use `--optimize <method>` in `planner_main.py`.
### Path Pruning (`path_pruning`)
Iteratively attempts to connect non-adjacent waypoints with collision-free straight lines to remove redundant waypoints and shorten the path.
- **Config**: `path_pruning.json`

### STOMP (`stomp`)
**Stochastic Trajectory Optimization for Motion Planning**. Generates noisy trajectory samples and iteratively updates the path to minimize a cost function combining smoothness and obstacle avoidance.
- **Config**: `stomp.json`
- **Reference**: Kalakrishnan et al., "STOMP: Stochastic trajectory optimization for motion planning" (ICRA 2011).

### GPMP2 (`gpmp2`)
**Gaussian Process Motion Planning 2**. Models the trajectory as a Gaussian Process and optimizes it using gradient-based methods (L-BFGS-B). Minimizes smoothness cost (GP prior) and obstacle cost.
- **Config**: `gpmp2.json`
- **Reference**: Dong et al., "Motion Planning as Probabilistic Inference using Gaussian Processes and Factor Graphs" (RSS 2016).

### TrajOpt (`trajopt`)
**Trajectory Optimization**. Uses Sequential Convex Optimization (via `scipy.optimize.minimize` SLSQP) to optimize a trajectory for smoothness while satisfying collision constraints.
- **Config**: `trajopt.json`
- **Reference**: Schulman et al., "Motion Planning with Sequential Convex Optimization and Convex Collision Checking" (IJRR 2014).

### TOPP-RA (`topp_ra`)
**Time-Optimal Path Parameterization**. Resamples the geometric path into a time-parameterized trajectory (uniform time steps) respecting velocity and acceleration limits. This implementation uses a simplified trapezoidal velocity profile generator.
- **Config**: `topp_ra.json`
- **Reference**: Pham et al., "A New Approach to Time-Optimal Path Parameterization based on Reachability Analysis" (TRO 2018).

### B-Spline (`bspline`)
**B-Spline Smoothing**. Fits a B-spline curve to the waypoints using `scipy.interpolate.splprep` and resamples it to generate a smooth path.
- **Config**: `bspline.json`
- **Reference**: Standard B-spline interpolation/approximation.

## 사용 방법

각 알고리즘은 `planner_main.py`를 통해 실행할 수 있습니다.

```bash
# Informed RRT* 실행 예시
python python/planner_main.py --algorithm informed_rrt_star --stl "sample/PIPE NO.1_fill.stl"

# Task-space RRT 실행 예시 (가중치 적용)
python python/planner_main.py --algorithm task_space_rrt --stl "sample/PIPE NO.1_fill.stl"
```
