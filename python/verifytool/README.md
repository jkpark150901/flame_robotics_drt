# Verify Positioner Tool

`verifytool`은 OptiTrack Motive/NatNet rigid body 데이터를 이용해 포지셔너의 이동 전후 위치 변화와 자세 변화를 측정하는 PyQt6 기반 GUI 도구입니다. 실행 진입점은 `python/verifypositioner.py`이며, 기본 설정은 `python/verifypositioner.cfg`를 사용합니다.

## 주요기능

Verify Positioner Tool은 Motive에서 송출되는 NatNet rigid body stream을 수신하여 최대 4개의 rigid body를 동시에 모니터링합니다. 사용자는 각 RB 슬롯을 켜고 원하는 RB ID를 지정할 수 있으며, 연결 후 현재 위치, 원점 대비 변화량, 거리 변화, 수신 FPS를 GUI에서 확인할 수 있습니다.

측정은 `Start`와 `Stop` 기반으로 동작합니다. `Start`를 누르면 선택된 각 rigid body의 현재 위치와 quaternion 자세를 시작 상태로 저장하고, 이후 들어오는 NatNet frame을 trajectory로 기록합니다. 포지셔너를 움직인 뒤 `Stop`을 누르면 종료 시점의 위치와 자세를 시작 시점과 비교하여 `dx`, `dy`, `dz`, 3차원 거리 변화, roll/pitch/yaw 변화, 회전각, 회전축을 계산합니다.

측정 중 기록된 trajectory는 3D plot으로 표시됩니다. 시작점과 종료점이 구분되어 보이고, 중간 frame의 좌표축 방향도 함께 표시되어 포지셔너의 이동 경로와 자세 변화를 시각적으로 확인할 수 있습니다. 표시 좌표계는 UI 시각화를 위해 `Left-hand Y-up` 변환을 적용합니다.

결과 저장 기능도 포함되어 있습니다. `Save`를 누르면 전체 결과가 `positioner_YYYYMMDD_HHMMSS.csv` 형태로 저장되고, trajectory가 기록된 rigid body는 `positioner_YYYYMMDD_HHMMSS_rb1_traj.csv` 같은 별도 CSV로 frame별 위치와 quaternion이 저장됩니다. 저장 위치의 기본값은 repo root입니다.

## 특징

이 도구는 로봇 SDK 없이 NatNet 데이터만으로 포지셔너 이동 검증을 수행하도록 구성되어 있습니다. 실제 import 기준의 핵심 의존성은 `PyQt6`, `numpy`, `scipy`, `matplotlib`, 그리고 저장소 내 `tools/NatNet/NatNetClient.py`입니다. Python 타입 힌트 문법상 Python 3.10 이상 사용을 권장합니다.

NatNet 수신은 GUI thread와 분리된 `QThread`에서 실행됩니다. 따라서 Motive stream을 수신하면서도 GUI의 버튼, 상태 표시, plot 갱신이 별도로 동작합니다. UI 갱신 signal은 rigid body별로 약 30 Hz 수준으로 제한되어 과도한 refresh를 피합니다.

네트워크 설정은 server/client IP를 분리해서 다룹니다. `Client` 값이 `auto`이면 server IP에 맞는 로컬 네트워크 인터페이스를 자동으로 선택합니다. server와 client가 loopback/비-loopback 조합으로 잘못 섞인 경우에는 연결 전에 오류를 발생시켜 잘못된 네트워크 조합을 빠르게 확인할 수 있습니다.

현재 `python/verifytool/verifypositioner.py`는 `.ui` 파일을 직접 `loadUi()`로 읽지 않고 코드에서 UI를 구성합니다. `python/verifytool/verifypositioner.ui`는 폴더 안에 남아 있지만 현재 실행 경로에서는 직접 사용되지 않습니다. 또한 `python/verifypositioner.cfg`의 `"gui": "verifytool.ui"` 값도 현재 실행에는 직접 반영되지 않습니다. 나중에 UI 로딩 방식으로 정리할 경우 실제 파일명인 `verifypositioner.ui`에 맞춰 설정을 갱신해야 합니다.

## 사용방법

저장소 루트에서 실행하는 것을 권장합니다.

```powershell
python python/verifypositioner.py
```

설정 파일을 직접 지정하려면 다음처럼 실행합니다.

```powershell
python python/verifypositioner.py --config python/verifypositioner.cfg
```

처음 실행 전에는 필요한 Python 패키지가 설치되어 있어야 합니다.

```powershell
pip install PyQt6 numpy scipy matplotlib
```

GUI가 열리면 Motive에서 rigid body streaming을 켠 뒤 `Server`에 Motive/NatNet 서버 IP를 입력합니다. `Client`는 보통 `auto`로 둡니다. 측정할 rigid body 슬롯을 체크하고 각 슬롯의 RB ID를 Motive의 rigid body ID에 맞춰 입력합니다. 최대 4개까지 동시에 선택할 수 있습니다.

`Connect`를 누르면 NatNet 수신이 시작됩니다. 연결 상태가 `connected`로 바뀌고 FPS와 live position이 갱신되는지 확인합니다. 정상적으로 데이터가 들어오면 `Start` 버튼을 눌러 시작 위치와 자세를 기록합니다. 그 다음 포지셔너를 원하는 방식으로 움직이고, 이동이 끝난 뒤 `Stop`을 누릅니다.

`Stop` 이후 화면에는 각 RB의 위치 변화량, 거리 변화, 자세 변화, 회전축/회전각이 표시됩니다. 결과를 남기려면 `Save`를 누릅니다. 저장되는 메인 CSV에는 `rb`, `rb_id`, `timestamp_start`, `timestamp_stop`, `duration_s`, `dx_mm`, `dy_mm`, `dz_mm`, `droll_deg`, `dpitch_deg`, `dyaw_deg`, `angle_deg`, `axis_x`, `axis_y`, `axis_z`, eigenvalue 정보가 포함됩니다. trajectory CSV에는 frame별 timestamp, 위치 `x_m/y_m/z_m`, quaternion `qx/qy/qz/qw`가 저장됩니다.

필요한 파일만 다른 위치로 복사하려면 저장소 루트에서 다음 배치 파일을 사용할 수 있습니다.

```powershell
.\export_verifytool.bat D:\verifytool_export
```

`python` 폴더 안에서 실행 중이라면 wrapper를 사용할 수 있습니다.

```powershell
.\export_verifytool.bat D:\verifytool_export
```
