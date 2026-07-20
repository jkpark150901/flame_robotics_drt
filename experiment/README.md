# Experiment Commands

## Inspection IK Trace

Viewer에서 `Check IK`를 실행하면 IK 수렴 이력이 아래 폴더에 저장된다.

```text
experiment/inspection_ik/session_YYYYMMDD_HHMMSS/
  success/
  fallback/
  failed/
  collision/
```

파일명 형식:

```text
inspection_ik_YYYYMMDD_HHMMSS_<robot>_<normalized|raw>_<status>.csv
inspection_ik_YYYYMMDD_HHMMSS_<robot>_<normalized|raw>_<status>.json
```

CSV의 `iteration=0`은 initial guess이고, `iteration=1`부터 least-squares update가 1회 적용된 결과다.

## Joint Convergence Plot

세션 폴더 전체를 시간순으로 찾아 joint/error plot을 생성한다.

```powershell
python experiment/inspection_ik_plot_joints.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS
```

특정 JSON 또는 CSV 하나만 plot:

```powershell
python experiment/inspection_ik_plot_joints.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS/success/inspection_ik_YYYYMMDD_HHMMSS_dda_rb10_1300e_raw_success.json
```

joint 값을 plot에서만 min-max normalize해서 모양 비교:

```powershell
python experiment/inspection_ik_plot_joints.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS --normalized-view
```

출력 폴더 지정:

```powershell
python experiment/inspection_ik_plot_joints.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS -o experiment/inspection_ik/session_YYYYMMDD_HHMMSS/my_plots
```

## URDF Mesh Replay

IK trace를 URDF mesh와 함께 재생한다. JSON 또는 같은 stem의 JSON이 있는 CSV를 입력하면 URDF/base pose/joint names를 자동으로 읽는다.

```powershell
python experiment/inspection_ik_visualize.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS/success/inspection_ik_YYYYMMDD_HHMMSS_dda_rb10_1300e_raw_success.json
```

CSV 직접 입력:

```powershell
python experiment/inspection_ik_visualize.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS/success/inspection_ik_YYYYMMDD_HHMMSS_dda_rb10_1300e_raw_success.csv
```

특정 iteration frame만 보기:

```powershell
python experiment/inspection_ik_visualize.py <trace.json> --static --start-index 20
```

선택한 iteration 구간만 mesh로 재생:

```powershell
python experiment/inspection_ik_visualize.py <trace.json> --start-index 10 --end-index 80 --step 2 --delay 0.03
```

재생 속도 조절:

```powershell
python experiment/inspection_ik_visualize.py <trace.json> --step 5 --delay 0.05
```

JSON 없이 CSV만 있을 때 URDF를 직접 지정:

```powershell
python experiment/inspection_ik_visualize.py trace.csv --urdf urdf/rb10_1300e_DDA.urdf --robot-name dda_rb10_1300e --target-link dda_link_end
```

## Notebook Plot

iteration 구간을 슬라이더로 골라 joint/error graph를 확인한다.

```powershell
jupyter notebook experiment/inspection_ik_plot_notebook.ipynb
```

노트북에서 `root/file`에 세션 폴더 또는 trace 파일을 입력한 뒤 `Reload traces`를 누른다.

노트북 추가 기능:

- 선택 구간 joint angle plot: revolute joint는 degree, linear/track/carriage joint는 raw unit으로 분리 표시
- 선택 구간 mesh replay command 출력
- 선택 구간 mesh replay 바로 실행

## Pink IK Compare

저장된 IK trace와 같은 URDF, target pose, initial q를 사용해서 `pink` IK 결과와 비교한다.

```powershell
python experiment/inspection_ik_compare_pink.py experiment/inspection_ik/session_YYYYMMDD_HHMMSS/fallback/inspection_ik_YYYYMMDD_HHMMSS_dda_rb10_1300e_raw_fallback.json
```

결과는 trace 폴더의 `pink_compare/` 아래에 저장된다.

```text
pink_compare/
  <trace>_pink.csv
  <trace>_pink_compare.png
```

solver나 반복 횟수 변경:

```powershell
python experiment/inspection_ik_compare_pink.py <trace.json> --solver quadprog --max-iter 1000 --dt 0.35
```

`pink` 또는 QP solver가 없으면 설치가 필요하다.

```powershell
pip uninstall -y pink
pip install pin-pink qpsolvers quadprog
```

주의: `pip install pink`로 설치되는 패키지는 로봇 IK용 Pink가 아니라 코드 포매터 패키지일 수 있다. 이 경우 `site-packages/pink.py`가 import되어 `from pink.tasks import FrameTask`가 실패한다.
