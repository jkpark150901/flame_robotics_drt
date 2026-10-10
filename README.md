# flame_robotics_drt

## Setup

## Setup Environments (Python 3.10)

1. Install python packages
```
$ pip install -r requirements.txt
```

If you use conda or a custom Python environment, set the interpreter path in
`Makefile` or define `PYTHON_CONFIG` before calling `run.bat`.

```
PYTHON_CONFIG := C:/Users/admin/miniforge3/envs/drt/python.exe
```

```
set "PYTHON_CONFIG=C:\Users\admin\miniforge3\envs\drt\python.exe"
```

## Runtime Processes

Inspection workflow ownership is separated as follows.

| Process | Entry point | Responsibility |
| --- | --- | --- |
| SimTool | `python/simtool.py` | Owns selected points, returned poses, pose groups, planner results, and playback requests |
| Viewer | `python/viewervedo.py` | Point picking and visualization of pose, planning, and playback results |
| Robot Core | `python/robotcore.py` | Executes pose determination, IK, path planning, and returns waypoints |
| ZProxy | `python/zproxy.py` | Routes application ZAPI messages |
| Controller | `python/controller.py` | Optional real-device control process |

The current application uses `viewervedo`. `python/viewero3d/visualizer.py` is not
the active inspection-planning Viewer.

Data flow:

```text
SimTool -- ZAPI --> Viewer -- scene snapshot/request --> Robot Core
SimTool <-- ZAPI -- Viewer <-- pose/waypoints/result -- Robot Core
```

The Viewer only renders Robot Core results in its main process. SimTool keeps the
authoritative workflow state and sends the stored plan sequence back to the
Viewer when playback is requested.

Inspection planning is split across two non-Viewer handlers:

- `python/simtool/inspection_path_handler.py` builds requests and owns returned
  planning/playback state.
- `python/robot_core/path_planning_service.py` executes robot jobs and planner
  batches inside Robot Core.

## Run Modes

### Standalone Robot Core

Recommended when the Robot Core should be inspected, restarted, or benchmarked
independently. The endpoint defaults to `tcp://127.0.0.1:5557` from
`python/viewervedo.cfg`.

```bat
run.bat robotcore
run.bat viewer --robot_core_mode external
run.bat simtool
```

The complete standalone topology can be started with:

```bat
run.bat run
```

This starts ZProxy, Robot Core, and Viewer in separate terminals. SimTool remains
in the foreground.

### Embedded Robot Core

In embedded mode the Viewer creates and terminates a child Robot Core process.

```bat
run.bat run-embedded
```

Equivalent individual commands are:

```bat
run.bat zproxy
run.bat viewer --robot_core_mode embedded
run.bat simtool
```

The `robot_core_service.mode` setting in `python/viewervedo.cfg` is used when
`--robot_core_mode` is omitted.

## Process Shutdown

- Close SimTool and Viewer normally through their windows.
- Press `F12` in the Viewer to terminate its embedded Robot Core or request
  shutdown of the connected standalone Robot Core. The key is configurable with
  `robot_core_service.shutdown_hotkey`.
- Use `Ctrl+C` in the standalone Robot Core terminal for direct shutdown.
- ZProxy is a separate process and must be closed from its terminal when it was
  started separately or by `run.bat run`.

## Individual Commands

Additional arguments after a target are forwarded to that Python entry point.

```bat
run.bat viewer --verbose_level DEBUG
run.bat robotcore --endpoint tcp://127.0.0.1:5557
run.bat simtool --verbose_level INFO
run.bat controller
```

Ubuntu targets remain available through the existing `Makefile`, for example
`make viewer` and `make simtool`.



python python/apf_heatmap.py \
    --config python/viewervedo.cfg \
    --snapshot sample/planning3.pkl \
    --joint-states-csv debug/RRTConnect_20260810_213301/14_dda_rb10_1300e_DDA/joint_states.csv \
    --robot-name dda_rb10_1300e \
    --waypoint 4 \
    --rotate \
    --apf-field-joints dda_joint_linear_track dda_joint_carriage \
    --apf-field-range 0.05 0.5 \
    --apf-field-steps 40 \
    --apf-d0 0.5 --apf-eta 1.0 --apf-k-att 1.0