"""tkinter GUI: 설정 -> 실행(시뮬레이션/실기) -> 결과 확인. 표준 라이브러리 tkinter + matplotlib 만 사용한다."""
import csv
import glob
import json
import os
import platform
import queue
import subprocess
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import matplotlib.image as mpimg
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import __version__, analysis, meshview, trajectory
from .mock_robot import MockRobot
from .runner import Runner
from .settings import Settings
from .time_axis import TimeAxisControls

# (필드명, 라벨, 종류[, 선택지]) — 종류: str/float/int/bool/choice/file/dir/list6
FORM = {
    "연결/궤적": [
        ("ip", "제어박스 IP", "str"),
        ("use_mock", "로봇 없이 테스트 (내장 목업 127.0.0.1)", "bool"),
        ("robot_mode", "로봇 모드", "choice", ["simulation", "real"]),
        ("real_speed_cap", "실기 속도 상한 [deg/s] (0=없음)", "float"),
        ("revert_to_sim", "실기 종료 후 simulation 으로 복귀", "bool"),
        ("traj_path", "궤적 파일(CSV)", "file"),
        ("urdf_path", "URDF (링크 표면 오차용, 비우면 계산 안 함)", "file"),
        ("surface_points", "링크당 표면 샘플점 수", "int"),
        ("surface_samples", "표면 비교에 쓸 매칭쌍 수", "int"),
        ("traj_kind", "궤적 형식", "choice", ["waypoint", "interp"]),
        ("vmax", "웨이포인트 시간 부여 속도 [deg/s]", "float"),
        ("interp_rate", "보간 주파수 [Hz]", "float"),
    ],
    "실행/모션": [
        ("mode", "추종 방식", "choice", ["jb2", "servo", "both"]),
        ("step", "기본 추출 간격 (극값 추가 보존)", "int"),
        ("out_dir", "출력 폴더", "dir"),
        ("home_first", "시작점 이동 전에 원점 복귀", "bool"),
        ("home_joints", "원점 관절각 6개 [deg] (쉼표)", "list6"),
        ("home_speed", "원점 복귀 속도 [deg/s]", "float"),
        ("home_accel", "원점 복귀 가속도 [deg/s²]", "float"),
        ("speed", "move_j / jb2 / servo 프로파일 속도 [deg/s]", "float"),
        ("accel", "move_j / jb2 / servo 프로파일 가속도 [deg/s²]", "float"),
        ("blend", "jb2 블렌드율 [%]", "float"),
        ("servo_timing", "servo 타이밍 (profile=속도/가속도 프로파일)", "choice", ["profile", "original"]),
        ("servo_basis", "servo 프로파일 한계 기준", "choice", ["max_joint", "l2"]),
        ("servo_period", "servo 전송 주기 [s] (profile, 0.002~0.2)", "float"),
        ("t1", "servo t1 [s] (0=전송 주기)", "float"),
        ("t2", "servo t2 [s] (0=자동: profile 주기×2, 0.02~0.2)", "float"),
        ("gain", "servo gain", "float"),
        ("alpha", "servo alpha (0~1)", "float"),
        ("final_t2", "servo 마지막 명령의 t2 [s] (0=t2와 같음, 최소 0.02)", "float"),
        ("final_hold", "servo 마지막 점 반복 횟수 (종료 오버슈트 억제, 0=없음)", "int"),
    ],
    "대기/기록": [
        ("reach_tol", "도달 허용 오차 [deg]", "float"),
        ("reach_hold", "도달 유지 시간 [s]", "float"),
        ("timeout", "움직임 없을 때 포기 시간 [s]", "float"),
        ("settle_timeout", "servo 수렴 대기 포기 시간 [s]", "float"),
        ("max_wait", "대기 절대 상한 [s]", "float"),
        ("record_hz", "상태 기록 주파수 [Hz]", "float"),
    ],
    "판정 기준": [
        ("extrema_prominence", "극값 검출 최소 돌출도 [deg]", "float"),
        ("extrema_max_shift", "극값 매칭 허용 시간 차이 (전체 대비)", "float"),
        ("th_path_p95", "궤적 추종 오차(DTW) p95 ≤ [deg]", "float"),
        ("th_path_max", "궤적 추종 오차(DTW) 최대 ≤ [deg]", "float"),
        ("th_cover", "커버리지 최대 ≤ [deg]", "float"),
        ("th_progress", "경로 진행도 ≥", "float"),
        ("th_step", "샘플 간 최대 이동 ≤ [deg]", "float"),
        ("th_end", "시작/종료 각 오차 ≤ [deg] (종료=마지막 안정 구간)", "float"),
        ("th_ripple", "속도 리플 ≤ (비율)", "float"),
        ("th_wobble", "횡방향 흔들림 ≤ [deg]", "float"),
        ("th_stall", "경로 중간 정체 비율 ≤", "float"),
        ("th_lag", "타이밍 지연 표시 기준 (비율)", "float"),
        ("move_thr", "모션 시작 판정 변화량 [deg]", "float"),
    ],
}


KR_PROPORTIONAL = ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "Apple SD Gothic Neo",
                   "UnDotum", "Dotum", "Gulim", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
KR_MONO = ["Noto Sans Mono CJK KR", "NanumGothicCoding", "D2Coding", "Noto Sans Mono CJK JP"]


def setup_fonts(root):
    """한글을 표시할 수 있는 글꼴을 찾아 모든 Tk 기본 글꼴에 적용한다. 설명 문자열을 반환."""
    fams = set(tkfont.families(root))
    prop = next((f for f in KR_PROPORTIONAL if f in fams), None)
    mono = next((f for f in KR_MONO if f in fams), prop)
    if prop is None:
        return ("한글 글꼴을 찾지 못함 (글꼴 %d개) -> 한글이 깨질 수 있습니다. 'sudo apt install fonts-noto-cjk' 설치 후 다시 실행하세요. "
                "conda/ROS 환경이면 LD_LIBRARY_PATH 를 비우고 실행하세요 (run.sh 가 자동 처리)." % len(fams))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name).configure(family=prop)
        except tk.TclError:
            pass
    tkfont.nametofont("TkFixedFont").configure(family=mono)
    # matplotlib 그래프 안의 한글(안내 문구 등)도 같은 글꼴로
    try:
        import matplotlib
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        mpl_font = next((f for f in KR_PROPORTIONAL if f in have), None)
        if mpl_font:
            matplotlib.rcParams["font.sans-serif"] = [mpl_font] + list(matplotlib.rcParams["font.sans-serif"])
            matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        pass
    return f"한글 글꼴: {prop} / 고정폭: {mono}"


INTERACTIVE_PROFILE = "★ 위치/속도 프로파일 (관절 선택)"
INTERACTIVE_VELOCITY = "★ 속도 프로파일 (관절 선택)"
INTERACTIVE_MESH = "★ 메시 차이 (URDF, 시점 선택)"
PARAM_UNITS = {"speed": " deg/s", "accel": " deg/s²", "blend": " %", "t1": " s", "t2": " s"}


def open_path(path):
    if platform.system() == "Windows":
        os.startfile(path)  # noqa: S606
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class RealConfirm(tk.Toplevel):
    """실기 실행 확인 창: 사전 점검 정보를 보여주고, 체크박스 + 확인 문구 입력을 요구한다."""

    def __init__(self, master, info, s):
        super().__init__(master)
        self.title("실제 로봇 실행 확인")
        self.transient(master)
        self.grab_set()
        self.ok = False
        warn = tk.Label(self, text="⚠  실제 로봇이 움직입니다", fg="white", bg="#b00020", font=("", 13, "bold"), pady=6)
        warn.pack(fill="x")
        cur = ", ".join(f"{v:.1f}" for v in info["current"])
        st = ", ".join(f"{v:.1f}" for v in info["start_pose"])
        if info.get("home_only"):
            self.title("실제 로봇 원점 복귀 확인")
            lines = [
                f"대상: {info['ip']}   (현재 모드: {'simulation' if info['sim'] == 1 else 'REAL'}, robot_state={info['robot_state']})",
                f"현재 자세 [deg]: {cur}",
                f"원점 자세 [deg]: {st}",
                f"현재 → 원점 최대 관절 이동: {info['delta_to_start']:.1f}°",
                f"적용 속도/가속도: {info['speed']:g} deg/s, {info['accel']:g} deg/s²" + (f"  (상한 {s.real_speed_cap:g} 적용)" if s.real_speed_cap else "  (속도 상한 없음!)"),
                "동작: move_j 한 번 (관절 공간 직선 이동)",
            ]
        else:
            lines = [
            f"대상: {info['ip']}   (현재 모드: {'simulation' if info['sim'] == 1 else 'REAL'}, robot_state={info['robot_state']})",
            f"현재 자세 [deg]: {cur}",
            f"시작 자세 [deg]: {st}",
            f"현재 → 시작 자세 최대 관절 이동: {info['delta_to_start']:.1f}°" + (f"   (원점 복귀 최대 이동 {info['home_delta']:.1f}°)" if info["home_delta"] is not None else ""),
            f"추종 경로 길이(관절 최대): {info['path_deg']:.0f}°, 지령점 {info['n_points']}개",
            f"적용 속도/가속도: {info['speed']:g} deg/s, {info['accel']:g} deg/s²" + (f"  (상한 {s.real_speed_cap:g} 적용)" if s.real_speed_cap else "  (속도 상한 없음!)"),
            f"추종 방식: {s.mode}",
        ]
        tk.Label(self, text="\n".join(lines), justify="left", font=("TkFixedFont", 10), padx=14, pady=8).pack(anchor="w")
        tk.Label(self, justify="left", fg="#b00020", padx=14, text=(
            "• 이 프로그램의 [중지]는 task stop 시도일 뿐 비상정지가 아닙니다.\n"
            "• 반드시 펜던트/제어박스의 물리 E-Stop 에 손이 닿는 상태에서 실행하세요.\n"
            "• 로봇 주변에 사람/장애물이 없고, 경로가 충돌 없이 검증됐는지 확인하세요.")).pack(anchor="w")
        self.chk = tk.BooleanVar()
        ttk.Checkbutton(self, text="E-Stop 접근 가능, 주변 확인 완료", variable=self.chk, command=self._upd).pack(anchor="w", padx=14, pady=(8, 0))
        row = tk.Frame(self)
        row.pack(anchor="w", padx=14, pady=4)
        tk.Label(row, text="확인 문구 입력:  REAL  ").pack(side="left")
        self.txt = tk.StringVar()
        self.txt.trace_add("write", lambda *_: self._upd())
        ttk.Entry(row, textvariable=self.txt, width=10).pack(side="left")
        btns = tk.Frame(self)
        btns.pack(pady=10)
        self.go = ttk.Button(btns, text="실행", command=self._ok, state="disabled")
        self.go.pack(side="left", padx=8)
        ttk.Button(btns, text="취소", command=self.destroy).pack(side="left", padx=8)
        self.wait_window()

    def _upd(self):
        self.go.config(state="normal" if (self.chk.get() and self.txt.get().strip() == "REAL") else "disabled")

    def _ok(self):
        self.ok = True
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"RB Servo Tracking Tool v{__version__}")
        root.geometry("1400x860")
        self.font_info = setup_fonts(root)
        self.s = Settings.load()
        self.vars = {}
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.mock = None
        self.runs = {}  # tree iid -> (run_dir, json_path)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        self.tab_run = ttk.Frame(nb)
        self.tab_res = ttk.Frame(nb)
        nb.add(self.tab_run, text="  실행  ")
        nb.add(self.tab_res, text="  결과  ")
        self._build_run()
        self._build_results()
        self._load_into_form(self.s)
        self.refresh_runs()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll)
        if "찾지 못함" in self.font_info:
            self.log.insert("end", self.font_info + "\n")

    # ------------------------------------------------------------ 실행 탭
    def _build_run(self):
        left = ttk.Frame(self.tab_run)
        left.pack(side="left", fill="y", padx=6, pady=6)
        self.mode_banner = tk.Label(left, text="SIMULATION", bg="#2e7d32", fg="white", font=("", 12, "bold"), pady=4)
        self.mode_banner.pack(fill="x", pady=(0, 6))
        nb = ttk.Notebook(left)
        nb.pack(fill="y", expand=True)
        for name, fields_ in FORM.items():
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text=name)
            for r, spec in enumerate(fields_):
                self._add_field(f, r, spec)
        btn = ttk.Frame(left)
        btn.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(btn, text="▶ 실행", command=self.on_run)
        self.btn_run.pack(side="left", padx=2)
        self.btn_home = ttk.Button(btn, text="⌂ 원점 복귀", command=self.on_home)
        self.btn_home.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(btn, text="■ 중지 (task stop)", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=2)
        ttk.Button(btn, text="궤적 미리보기", command=self.on_preview).pack(side="left", padx=2)
        btn2 = ttk.Frame(left)
        btn2.pack(fill="x")
        ttk.Button(btn2, text="설정 저장…", command=self.on_save).pack(side="left", padx=2)
        ttk.Button(btn2, text="설정 불러오기…", command=self.on_load).pack(side="left", padx=2)
        self.status = tk.StringVar(value="대기 중")
        ttk.Label(left, textvariable=self.status, foreground="#555").pack(anchor="w", pady=4)

        right = ttk.Frame(self.tab_run)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.pfig = Figure(figsize=(7, 3.2))
        self.pcanvas = FigureCanvasTkAgg(self.pfig, master=right)
        self.ptoolbar = NavigationToolbar2Tk(self.pcanvas, right, pack_toolbar=False)
        self.ptoolbar.pack(fill="x")
        self.ptime = TimeAxisControls(right, self.pcanvas, self.ptoolbar)
        self.ptime.pack(fill="x")
        self.pcanvas.get_tk_widget().pack(fill="x")
        self.preview_info = tk.StringVar(value="궤적 파일을 선택하고 [궤적 미리보기]를 누르세요")
        ttk.Label(right, textvariable=self.preview_info).pack(anchor="w")
        self.log = tk.Text(right, height=18, wrap="none", font=("TkFixedFont", 9))
        sb = ttk.Scrollbar(right, command=self.log.yview)
        self.log.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True, pady=(6, 0))

    def _add_field(self, parent, row, spec):
        name, label, kind = spec[:3]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        if kind == "bool":
            v = tk.BooleanVar()
            ttk.Checkbutton(parent, variable=v).grid(row=row, column=1, sticky="w")
        elif kind == "choice":
            v = tk.StringVar()
            cb = ttk.Combobox(parent, textvariable=v, values=spec[3], state="readonly", width=14)
            cb.grid(row=row, column=1, sticky="w")
        else:
            v = tk.StringVar()
            w = 40 if kind in ("file", "dir") else (24 if kind == "list6" else 14)
            ttk.Entry(parent, textvariable=v, width=w).grid(row=row, column=1, sticky="w")
            if kind in ("file", "dir"):
                ttk.Button(parent, text="…", width=3, command=lambda n=name, k=kind: self._browse(n, k)).grid(row=row, column=2)
        self.vars[name] = (v, kind)
        if name == "robot_mode":
            v.trace_add("write", lambda *_: self._banner())

    def _browse(self, name, kind):
        v = self.vars[name][0]
        types = [("URDF", "*.urdf"), ("All", "*.*")] if name == "urdf_path" else [("CSV", "*.csv"), ("All", "*.*")]
        p = filedialog.askopenfilename(filetypes=types) if kind == "file" else filedialog.askdirectory()
        if p:
            v.set(p)

    def _banner(self):
        real = self.vars["robot_mode"][0].get() == "real"
        self.mode_banner.config(text="REAL — 실제 로봇" if real else "SIMULATION", bg="#b00020" if real else "#2e7d32")

    def _load_into_form(self, s):
        for name, (v, kind) in self.vars.items():
            val = getattr(s, name)
            if kind == "list6":
                v.set(", ".join(f"{x:g}" for x in val))
            elif kind == "bool":
                v.set(bool(val))
            else:
                v.set(str(val))
        self._banner()

    def _read_form(self):
        s = Settings()
        for name, (v, kind) in self.vars.items():
            raw = v.get()
            try:
                if kind == "bool":
                    val = bool(raw)
                elif kind == "int":
                    val = int(float(raw))
                elif kind == "float":
                    val = float(raw)
                elif kind == "list6":
                    val = [float(x) for x in str(raw).replace(";", ",").split(",") if x.strip()]
                    if len(val) != 6:
                        raise ValueError("6개 값이 필요")
                else:
                    val = str(raw).strip()
            except ValueError as e:
                label = next(sp[1] for fl in FORM.values() for sp in fl if sp[0] == name)
                raise ValueError(f"'{label}' 값이 올바르지 않습니다: {raw} ({e})") from e
            setattr(s, name, val)
        if s.step < 1:
            raise ValueError("서브샘플링 간격은 1 이상이어야 합니다")
        return s

    # ------------------------------------------------------------ 동작
    def logmsg(self, text):
        self.q.put(("log", text))

    def _poll(self):
        try:
            while True:
                kind, *payload = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload[0] + "\n")
                    self.log.see("end")
                elif kind == "done":
                    self._finish(*payload)
                elif kind == "mesh_series":
                    self._mesh_series_ready(*payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def on_preview(self):
        try:
            s = self._read_form()
            t, q = trajectory.prepare(s)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("궤적 오류", str(e))
            return
        import numpy as np
        t2, q2 = trajectory.subsample(t, np.rad2deg(q), s.step)
        self.pfig.clear()
        show_prof = s.mode in ("servo", "both") and s.servo_timing == "profile"
        ax = self.pfig.add_subplot(211 if show_prof else 111)
        for j in range(6):
            ax.plot(t, np.rad2deg(q[:, j]), lw=1.2, label=f"j{j}")
        ax.plot(t2, np.full(len(t2), np.rad2deg(q).min()), "|", color="k", ms=6)  # 지령점 눈금
        ax.set_xlabel("time [s]")
        ax.set_ylabel("angle [deg]")
        ax.grid(alpha=0.3)
        ax.legend(ncol=6, fontsize=7, frameon=False)
        prof_txt = ""
        if show_prof:
            try:
                prof = trajectory.PathProfile.fit(np.rad2deg(q), s.speed, s.accel, s.servo_basis, s.servo_period)
                tg, send_t, iv = prof.schedule(s.servo_period)
            except ValueError as e:
                messagebox.showerror("servo 프로파일 오류", str(e))
                return
            tt = np.linspace(0, prof.T, max(int(prof.T * 100), 2) + 1)
            _, vv = prof.joints(tt)
            ax2 = self.pfig.add_subplot(212)
            for j in range(6):
                ax2.plot(tt, vv[:, j], ":", lw=1.4, color=f"C{j}")
            ax2.plot(prof.t, prof.v, color="k", lw=1.0, alpha=0.6)  # 경로 속력 (basis 기준)
            ax2.set_xlabel("servo profile time [s]   (dotted = ideal joint velocity, gray = path speed)")
            ax2.set_ylabel("deg/s")
            ax2.grid(alpha=0.3)
            prof_txt = (f"  |  servo 이상 프로파일 {prof.T:.1f}s, 최고 {prof.vp:.1f} deg/s ({s.servo_basis}), "
                        f"전송 {len(tg)}개 × {iv[0] * 1000:.1f} ms, 지령 차분 최대 속도 {prof.cmd_speed_max:.1f} / 가속도 {prof.cmd_accel_max:.1f}")
        self.pfig.tight_layout()
        self.ptime.refresh((s.traj_path, s.traj_kind, s.vmax, s.interp_rate, s.step, s.mode, s.servo_timing, s.speed, s.accel, s.servo_basis, s.servo_period))
        self.pcanvas.draw()
        self.preview_info.set(f"기준 궤적 {len(t)}점 @ {1 / (t[1] - t[0]):.0f}Hz, {t[-1] - t[0]:.1f}s  →  지령 {len(t2)}점 (극값 보존, 간격 {np.diff(t2).min() * 1000:.1f}~{np.diff(t2).max() * 1000:.1f} ms)  [아래 눈금 = 지령점]" + prof_txt)

    def _ready(self, home=False):
        """실행/원점 복귀 공통 준비: 설정 읽기, 목업 시작, real 이면 사전 점검 + 확인 창. 진행해도 되면 Settings, 아니면 None."""
        try:
            s = self._read_form()
        except ValueError as e:
            messagebox.showerror("설정 오류", str(e))
            return None
        if s.use_mock and self.mock is None:
            try:
                self.mock = MockRobot(vmax=120.0).start()
                self.logmsg("내장 목업 로봇 시작 (127.0.0.1:5000/5001)")
            except OSError as e:
                messagebox.showerror("목업 시작 실패", f"{e}\n(포트 5000/5001 이 이미 사용 중일 수 있습니다)")
                return None
        if s.robot_mode == "real":
            if s.use_mock:
                messagebox.showwarning("모드 충돌", "목업 로봇과 real 모드는 함께 쓸 수 없습니다 (목업은 시험용)")
                return None
            self.status.set("실기 사전 점검 중…")
            self.root.update_idletasks()
            try:
                r = Runner(s, log=self.logmsg)
                info = r.preflight_home() if home else r.preflight()
            except Exception as e:  # noqa: BLE001
                self.status.set("대기 중")
                messagebox.showerror("사전 점검 실패", f"로봇에 연결/읽기 실패:\n{e}")
                return None
            self.status.set("대기 중")
            if info["robot_state"] != 1:
                messagebox.showerror("실행 불가", f"로봇이 정지 상태가 아닙니다 (robot_state={info['robot_state']}).")
                return None
            dlg = RealConfirm(self.root, info, s)
            if not dlg.ok:
                self.logmsg("실기 실행 취소됨")
                return None
            s.real_ack = True
        return s

    def _busy(self, on):
        st = "disabled" if on else "normal"
        self.btn_run.config(state=st)
        self.btn_home.config(state=st)
        self.btn_stop.config(state="normal" if on else "disabled")

    def on_home(self):
        """원점(home_joints)으로 복귀. 실행과 같은 안전장치를 적용한다."""
        if self.worker and self.worker.is_alive():
            return
        s = self._ready(home=True)
        if s is None:
            return
        self.stop.clear()
        self._busy(True)
        self.status.set("원점 복귀 중… (REAL)" if s.robot_mode == "real" else "원점 복귀 중…")
        self.log.delete("1.0", "end")

        def work():
            try:
                Runner(s, log=self.logmsg, stop=self.stop).home()
                self.q.put(("done", None, None))
            except Exception as e:  # noqa: BLE001
                self.q.put(("done", None, e))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def on_run(self):
        if self.worker and self.worker.is_alive():
            return
        s = self._ready()
        if s is None:
            return
        self.stop.clear()
        self.log.delete("1.0", "end")
        self._busy(True)
        self.status.set("실행 중… (REAL)" if s.robot_mode == "real" else "실행 중…")
        try:
            s.save()
        except OSError:
            pass
        self.worker = threading.Thread(target=self._work, args=(s,), daemon=True)
        self.worker.start()

    def _work(self, s):
        r = Runner(s, log=self.logmsg, stop=self.stop)
        try:
            d = r.run()
            self.q.put(("done", d, None))
        except Exception as e:  # noqa: BLE001
            self.q.put(("done", r.run_dir, e))

    def _finish(self, run_dir, err):
        self._busy(False)
        self.status.set(f"중단: {err}" if err else (f"완료: {run_dir}" if run_dir else "원점 복귀 완료"))
        if run_dir:
            self.refresh_runs(select_dir=run_dir)

    def on_stop(self):
        self.stop.set()
        self.logmsg("중지 요청됨 (task stop 전송 시도). 비상정지가 아니므로 필요하면 물리 E-Stop 을 사용하세요.")

    def on_save(self):
        try:
            s = self._read_form()
        except ValueError as e:
            messagebox.showerror("설정 오류", str(e))
            return
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            s.save(p)

    def on_load(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if p:
            self._load_into_form(Settings.load(p))

    # ------------------------------------------------------------ 결과 탭
    def _build_results(self):
        # 표/상세(왼쪽)와 그래프(오른쪽) 사이의 경계를 끌어서 폭을 조절할 수 있다
        pw = ttk.PanedWindow(self.tab_res, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=1)
        pw.add(right, weight=3)
        self._pw, self._sash_done = pw, False
        self.tab_res.bind("<Map>", lambda e: self.root.after(50, self._init_sash))
        cols = ("run", "mode", "step", "t_track", "t_total", "p95", "end", "ripple", "wobble", "rough", "surf", "surf95", "pass")
        heads = ("실행", "방식", "step", "추종[s]", "지령→완료[s]", "추종오차p95", "종료각오차", "속도리플", "흔들림[deg]", "거친구간", "표면최악[mm]", "표면p95[mm]", "판정")
        widths = (225, 50, 42, 62, 100, 82, 78, 70, 92, 72, 90, 90, 55)
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=11, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center" if c != "run" else "w")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self.on_select_run)
        b = ttk.Frame(left)
        b.pack(fill="x", pady=6)
        ttk.Button(b, text="새로고침", command=self.refresh_runs).pack(side="left", padx=2)
        ttk.Button(b, text="다시 분석", command=self.on_reanalyze).pack(side="left", padx=2)
        ttk.Button(b, text="폴더 열기", command=self.on_open_dir).pack(side="left", padx=2)
        ttk.Button(b, text="summary.csv", command=self.on_open_summary).pack(side="left", padx=2)
        dframe = ttk.Frame(left)
        dframe.pack(fill="both", expand=True)
        self.detail = tk.Text(dframe, height=16, width=92, font=("TkFixedFont", 9))
        dsb = ttk.Scrollbar(dframe, command=self.detail.yview)
        self.detail.config(yscrollcommand=dsb.set)
        dsb.pack(side="right", fill="y")
        self.detail.pack(side="left", fill="both", expand=True)

        top = ttk.Frame(right)
        top.pack(fill="x")
        self._top = top
        ttk.Label(top, text="그래프:").pack(side="left")
        self.img_var = tk.StringVar()
        self.img_cb = ttk.Combobox(top, textvariable=self.img_var, state="readonly", width=70)
        self.img_cb.pack(side="left", padx=6)
        self.img_cb.bind("<<ComboboxSelected>>", lambda e: self.show_image())
        # 관절 선택 (대화형 그래프 '★ ...' 에서만 표시)
        self.joint_frame = ttk.Frame(right)
        ttk.Label(self.joint_frame, text="관절:").pack(side="left")
        self.jvars = []
        for j in range(6):
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.joint_frame, text=f"J{j + 1}", variable=v, command=self.show_interactive).pack(side="left", padx=3)
            self.jvars.append(v)
        ttk.Button(self.joint_frame, text="전체", width=5, command=lambda: self._set_joints(True)).pack(side="left", padx=(10, 2))
        ttk.Button(self.joint_frame, text="해제", width=5, command=lambda: self._set_joints(False)).pack(side="left", padx=2)
        self.abs_var = tk.BooleanVar(value=True)  # 속도를 절대값(크기)로 볼지, 부호 있는 속도로 볼지
        ttk.Checkbutton(self.joint_frame, text="속도 절대값", variable=self.abs_var, command=self.show_interactive).pack(side="left", padx=(14, 2))
        self._prof = {}
        self.cur_mode = None
        # 메시 차이 보기 컨트롤 ('★ 메시 차이' 에서만 표시)
        self.mesh_frame = ttk.Frame(right)
        r1 = ttk.Frame(self.mesh_frame); r1.pack(fill="x")
        ttk.Label(r1, text="시점: 위 그래프를 클릭/드래그(놓으면 갱신)").pack(side="left")
        ttk.Button(r1, text="최악", width=5, command=lambda: self._mesh_jump("worst")).pack(side="left", padx=(8, 2))
        ttk.Button(r1, text="처음", width=5, command=lambda: self._mesh_jump("first")).pack(side="left", padx=2)
        ttk.Button(r1, text="끝", width=4, command=lambda: self._mesh_jump("last")).pack(side="left", padx=2)
        self.mv_tlabel = tk.StringVar(value="")
        ttk.Label(r1, textvariable=self.mv_tlabel, foreground="#b00020").pack(side="left", padx=8)
        r2 = ttk.Frame(self.mesh_frame); r2.pack(fill="x", pady=2)
        self.mv_az, self.mv_el = tk.DoubleVar(value=40), tk.DoubleVar(value=22)
        for text, var, lo, hi, w in (("방위", self.mv_az, -180, 180, 100), ("고도", self.mv_el, -10, 90, 80)):
            ttk.Label(r2, text=text).pack(side="left")
            sc = ttk.Scale(r2, from_=lo, to=hi, variable=var, orient="horizontal", length=w, command=lambda v: self._mesh_scale_label())
            sc.pack(side="left", padx=(2, 8)); sc.bind("<ButtonRelease-1>", lambda e: self._mesh_render())
        self.mv_val = tk.StringVar(value="")
        ttk.Label(r2, textvariable=self.mv_val, foreground="#555").pack(side="left")
        r3 = ttk.Frame(self.mesh_frame); r3.pack(fill="x")
        for text, az, el in (("정면", -90, 5), ("측면", 0, 5), ("위", 0, 89), ("등각", 40, 22)):
            ttk.Button(r3, text=text, width=5, command=lambda a=az, e=el: self._mesh_view(a, e)).pack(side="left", padx=2)
        ttk.Label(r3, text="  링크").pack(side="left")
        self.mv_focus = tk.StringVar(value="전체")
        self.mv_focus_cb = ttk.Combobox(r3, textvariable=self.mv_focus, state="readonly", width=9, values=["전체"])
        self.mv_focus_cb.pack(side="left", padx=2)
        self.mv_focus_cb.bind("<<ComboboxSelected>>", lambda e: self._mesh_render())
        self.mv_ref = tk.BooleanVar(value=True); self.mv_fixed = tk.BooleanVar(value=True); self.mv_struct = tk.BooleanVar(value=False)
        ttk.Checkbutton(r3, text="기준 자세 표시", variable=self.mv_ref, command=self._mesh_render).pack(side="left", padx=(8, 2))
        ttk.Checkbutton(r3, text="색 범위 고정(전체 최대)", variable=self.mv_fixed, command=self._mesh_render).pack(side="left", padx=2)
        ttk.Checkbutton(r3, text="레일 등 구조물 표시", variable=self.mv_struct, command=self._mesh_render).pack(side="left", padx=2)
        self._mesh = None; self._mesh_active = False; self._mesh_drag = False
        self._scene_cache = {}
        self._ax_time = self._ax_time2 = self._ax_mesh = self._cax = self._cursor = None
        self.rfig = Figure(figsize=(8, 8))
        self.rcanvas = FigureCanvasTkAgg(self.rfig, master=right)
        self.rtoolbar = NavigationToolbar2Tk(self.rcanvas, right)
        self.rtoolbar.update()
        self.rtime = TimeAxisControls(right, self.rcanvas, self.rtoolbar)
        self.rtime.pack(fill="x")
        for ev in ("button_press_event", "motion_notify_event", "button_release_event"):
            self.rcanvas.mpl_connect(ev, self._mesh_mouse)
        self.rcanvas.get_tk_widget().pack(fill="both", expand=True)
        self.cur_dir = None

    def _init_sash(self):
        """결과 탭이 처음 표시될 때 표/그래프 경계를 한 번만 정한다 (탭이 숨겨져 있을 때 지정하면 0 이 된다)."""
        if self._sash_done:
            return
        w = self._pw.winfo_width()
        if w < 200:
            self.root.after(100, self._init_sash)
            return
        self._pw.sashpos(0, min(820, int(w * 0.56)))
        self._sash_done = True

    def _out_dir(self):
        try:
            return self.vars["out_dir"][0].get().strip() or self.s.out_dir
        except KeyError:
            return self.s.out_dir

    def refresh_runs(self, select_dir=None):
        self.tree.delete(*self.tree.get_children())
        self.runs.clear()
        out = self._out_dir()
        sel_iid = None
        for d in sorted(glob.glob(os.path.join(out, "*_step*")), reverse=True):
            if not os.path.isdir(d):
                continue
            js = sorted(glob.glob(os.path.join(d, "compare_*.json")))
            name = os.path.basename(d)
            if not js:
                iid = self.tree.insert("", "end", values=(name, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "분석없음"))
                self.runs[iid] = (d, None)
            for j in js:
                with open(j, encoding="utf-8") as f:
                    r = json.load(f)
                M = r["metrics"]
                fm = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
                pc = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
                wb = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
                iid = self.tree.insert("", "end", values=(name, r["mode"], r["step"], fm(M.get("t_track")), fm(M.get("t_total")),
                                                          f"{M['path_dev_p95']:.2f}", f"{M['end_err']:.2f}",
                                                          pc(M.get("smooth_ripple")), wb(M.get("smooth_wobble")), pc(M.get("rough_ratio")),
                                                          fm(M.get("surf_disp_max")), fm(M.get("surf_disp_p95")),
                                                          "PASS" if r.get("all_pass") else "FAIL"))
                self.runs[iid] = (d, j)
        if select_dir:
            for iid, (d, _) in self.runs.items():
                if os.path.abspath(d) == os.path.abspath(select_dir):
                    sel_iid = iid
                    break
        if sel_iid:
            self.tree.selection_set(sel_iid)
            self.tree.see(sel_iid)

    def _run_header(self, d, r):
        """선택한 실행의 방식(method)·파라미터와 실행 설정을 읽기 쉬운 줄 목록으로 만든다 (config.json + 분석 JSON)."""
        try:
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                c = json.load(f)
        except (OSError, ValueError):
            c = {}
        eff = c.get("effective", c)
        req = c.get("requested", eff)
        g = lambda k, dflt="-": eff.get(k, dflt)  # noqa: E731
        lines = []
        mode = (r or {}).get("mode") or g("mode", "")
        methods = {mode: r["params"]} if r else {}
        if not methods and eff:
            if g("mode") in ("jb2", "both"):
                methods["jb2"] = {"method": "move_jb2_add", "speed": g("speed"), "accel": g("accel"), "blend": g("blend")}
            if g("mode") in ("servo", "both"):
                methods["servo"] = {"method": "move_servo_j", "t1": g("t1"), "t2": g("t2"), "gain": g("gain"), "alpha": g("alpha")}
        for m, mp in methods.items():
            lines.append(f"[방식] {mp.get('method', m)}   (모드: {m})")
            for k, v in mp.items():
                if k == "method":
                    continue
                txt = f"{v:g}" if isinstance(v, (int, float)) else str(v)
                txt += PARAM_UNITS.get(k, "")
                rq = req.get(k)
                if k in ("speed", "accel") and isinstance(rq, (int, float)) and isinstance(v, (int, float)) and abs(rq - v) > 1e-9:
                    txt += f"   (요청 {rq:g} → 실기 속도 상한 적용)"
                if k in ("t1", "t2") and rq == 0:
                    txt = "전송 주기 × 2" if v == "2x_period" else txt + "   (자동)"
                lines.append(f"    {k:<6} = {txt}")
        real = g("robot_mode", "simulation") == "real"
        lines.append(f"[실행] 로봇 모드 {'REAL (실제 로봇)' if real else 'simulation'}  |  기본 추출 1/{g('step')} + 극값 보존  |  기록 {g('record_hz')} Hz")
        lines.append(f"    궤적: {os.path.basename(str(g('traj_path', '')))} ({g('traj_kind')}, 시간 부여 {g('vmax')} deg/s, 보간 {g('interp_rate')} Hz)")
        if "speed" in eff:
            lines.append(f"    시작 자세 이동(move_j): {g('speed')} deg/s, {g('accel')} deg/s²")
        if g("home_first", False):
            lines.append(f"    원점 복귀: 사용 (위치 {g('home_joints')}, {g('home_speed')} deg/s)")
        else:
            lines.append("    원점 복귀: 안 함")
        lines.append(f"    대기: 도달 허용 {g('reach_tol')} deg / 유지 {g('reach_hold')} s / 무동작 포기 {g('timeout')} s")
        return lines

    def on_select_run(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        d, j = self.runs[sel[0]]
        self.cur_dir = d
        # 표시하는 그래프는 3종만: sim_follow_*, smooth_map_follow_*, compare_follow_*_after (이전 실행에 남은 다른 PNG 는 숨김)
        pngs = [p for p in sorted(glob.glob(os.path.join(d, "*.png")))
                if os.path.basename(p).startswith(("sim_follow", "smooth_map_follow"))
                or (os.path.basename(p).startswith("compare_follow") and p.endswith("_after.png"))]
        m = self.tree.item(sel[0], "values")[1]
        self.cur_mode = m
        order = sorted(pngs, key=lambda p: (0 if f"_{m}_after" in p else 1 if "smooth_map_" in p and f"_{m}." in p else 2 if "sim_" in os.path.basename(p) else 3, p))
        inter = [INTERACTIVE_MESH, INTERACTIVE_PROFILE, INTERACTIVE_VELOCITY] if j else []
        self.img_cb["values"] = inter + [os.path.basename(p) for p in order]
        if order:
            self.img_var.set(os.path.basename(order[0]))
            self.show_image()
        else:
            self.rfig.clear()
            self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
            self.rcanvas.draw()
        self.detail.delete("1.0", "end")
        if j:
            with open(j, encoding="utf-8") as f:
                r = json.load(f)
            M, V = r["metrics"], r["verdict"]
            fm = lambda v: "-" if v is None else f"{v:.2f}s"  # noqa: E731
            g = lambda k, f="{:.2f}": "-" if M.get(k) is None else f.format(M[k])  # noqa: E731  (없는 키는 '-')
            lines = [os.path.basename(d), *self._run_header(d, r), "",
                     f"추종 시간: 지령→시작 {fm(M.get('t_latency'))} / 추종 {fm(M.get('t_track'))} (원본 {fm(M.get('t_ref'))}, x{M.get('track_ratio', 0):.2f})"
                     f" / 지령→완료 {fm(M.get('t_total'))}",
                     f"추종 속도: 관절 최대 {g('v_max_joint', '{:.1f}')} deg/s, 속력 최대 {g('v_max_norm', '{:.1f}')} / 평균 {g('v_mean_norm', '{:.1f}')} deg/s",
                     f"매끄러움: 속도 리플 {g('smooth_ripple', '{:.1%}')}, 횡방향 흔들림 {g('smooth_wobble', '{:.3f}')} deg, 정체 {g('smooth_stall', '{:.1%}')}, 기준 초과 구간 {g('rough_ratio', '{:.0%}')} (최대 점수 {g('rough_score_max')})",
                     f"궤적 추종 오차(DTW) p95 {M['path_dev_p95']:.2f} / max {M['path_dev_max']:.2f} deg, 커버리지 max {M['cover_max']:.2f} deg",
                     f"진행도 {M['progress_max']:.1%}, 시작 {M['start_err']:.2f} / 종료 {M['end_err']:.2f} deg",
                     *([f"링크 표면 오차: 최악 점 변위 최대 {M['surf_disp_max']:.1f} mm / p95 {M['surf_disp_p95']:.1f} / 평균 {M['surf_disp_mean']:.1f} mm, "
                        f"Chamfer 최대 {M['surf_chamfer_max']:.2f} mm, Hausdorff 최대 {M['surf_hausdorff_max']:.1f} mm (최악 링크 {M['surf_worst_link']})"]
                       if M.get("surf_disp_max") is not None else []),
                     f"타이밍 지연 {M['lag_min']:+.3f} ~ {M['lag_max']:+.3f} (+늦음/-빠름)",
                     ""] + [f"{'PASS' if v else 'FAIL'}  {k}" for k, v in V.items()]
            match = r.get("matching", {})
            if match:
                lines.extend(["", "극값 기준 매칭 (양수 deficit=미도달, 음수=초과)",
                              f"기준점 {M.get('extrema_anchor_pairs',0)}쌍 / 누락 {M.get('extrema_missing',0)}개 / 순서 충돌 {M.get('extrema_order_conflicts',0)}개"])
                for event in match.get('extrema', []):
                    label = f"J{event['joint']} {event['kind']}"
                    if event['error_deg'] is None:
                        lines.append(f"  {label}: {event['status']}")
                    else:
                        lines.append(f"  {label}: ref {event['ref_time_s']:.3f}s / meas {event['meas_time_s']:.3f}s, "
                                     f"{event['ref_deg']:.3f} → {event['meas_deg']:.3f}°, "
                                     f"미도달 {event['deficit_deg']:+.3f}° [{event['status']}]")
            self.detail.insert("end", "\n".join(lines))
        else:
            self.detail.insert("end", "\n".join([os.path.basename(d), *self._run_header(d, None), "", "(분석 결과 없음 — 중단되었거나 분석 전 실행)"]))

    def _set_joints(self, on):
        for v in self.jvars:
            v.set(on)
        self.show_interactive()

    def show_interactive(self):
        """'★' 항목: 실행 폴더 데이터로 선택한 관절만 화면에서 직접 다시 그린다."""
        kind = self.img_var.get()
        if not self.cur_dir or not kind.startswith("★"):
            return
        joints = [j for j, v in enumerate(self.jvars) if v.get()]
        self.rfig.clear()
        if not joints:
            self.rfig.text(0.5, 0.5, "관절을 하나 이상 선택하세요", ha="center", va="center")
            self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
            self.rcanvas.draw()
            return
        key = (self.cur_dir, self.cur_mode)
        try:
            if key not in self._prof:
                self._prof = {key: analysis.profile_data(self.cur_dir, self.cur_mode)}
            d = self._prof[key]
        except Exception as e:  # noqa: BLE001  (오래된 실행은 일부 파일이 없을 수 있음)
            self.rfig.text(0.5, 0.5, f"그래프 데이터를 읽지 못했습니다\n{e}", ha="center", va="center")
            self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
            self.rcanvas.draw()
            return
        if kind == INTERACTIVE_PROFILE:
            analysis.fig_profile(d["head"], self.cur_mode, d["tr"], d["qr"], d["sub"], d["t_rel"], d["qw"], d["vel"],
                                 offset=d["offset"], joints=joints, fig=self.rfig, absolute=self.abs_var.get(), ideal=d.get("ideal"))
        else:
            analysis.fig_velocity(d["head"], d["t_rel"], d["vel"], d["speed_cmd"], joints=joints, fig=self.rfig, absolute=self.abs_var.get(),
                                  ideal=d.get("ideal"), accel_cmd=d.get("accel_cmd"), basis=d.get("basis", "max_joint"))
        self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
        self.rcanvas.draw()

    # ------------------------------------------------------------ 메시 차이 보기
    def _mesh_urdf(self):
        """(사용할 URDF, 분석에 기록된 URDF). 사용할 URDF 는 설정 탭의 현재 값 > 분석에 기록된 값 > 실행 설정 순.
        표면 오차 시계열(surface_*.csv)은 기록된 URDF 와 같을 때만 그대로 쓴다 (다르면 화면이 새로 계산)."""
        rec, cfg = "", ""
        try:
            js = sorted(glob.glob(os.path.join(self.cur_dir, f"compare_*_{self.cur_mode}.json")))
            with open(js[-1], encoding="utf-8") as f:
                rec = json.load(f).get("surface_urdf") or ""
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open(os.path.join(self.cur_dir, "config.json"), encoding="utf-8") as f:
                c = json.load(f)
            cfg = (c.get("effective", c) or {}).get("urdf_path", "") or ""
        except (OSError, ValueError):
            pass
        u = self.vars["urdf_path"][0].get().strip() or rec or cfg
        return (os.path.expanduser(u) if u else ""), rec

    def _mesh_text(self, msg):
        self.rfig.clear()
        self._mesh = None
        self.rfig.text(0.5, 0.5, msg, ha="center", va="center")
        self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
        self.rcanvas.draw()

    def show_mesh(self):
        d, m = self.cur_dir, self.cur_mode
        urdf, rec_urdf = self._mesh_urdf()
        if not urdf or not os.path.exists(urdf):
            self._mesh_text("URDF 파일이 필요합니다.\n설정 탭 '연결/궤적' 의 URDF 칸에 지정한 뒤 다시 선택하세요.")
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            if urdf not in self._scene_cache:
                self._scene_cache[urdf] = meshview.MeshScene(urdf)
            scene = self._scene_cache[urdf]
            pj = sorted(glob.glob(os.path.join(d, f"compare_*_{m}_pairs.csv")))[-1]
            with open(pj, encoding="utf-8") as f:
                f.readline()
                rows = list(csv.reader(f))[1:]
            t = np.array([float(x[0]) for x in rows])
            qm = np.deg2rad(np.array([[float(v) for v in x[4:10]] for x in rows]))
            qr = np.deg2rad(np.array([[float(v) for v in x[10:16]] for x in rows]))
            dev = np.array([float(x[22]) for x in rows])
        except Exception as e:  # noqa: BLE001
            self._mesh_text(f"메시 데이터를 읽지 못했습니다\n{e}")
            return
        finally:
            self.root.config(cursor="")
        series = None
        sf = sorted(glob.glob(os.path.join(d, f"surface_*_{m}.csv")))
        same = bool(rec_urdf) and os.path.abspath(rec_urdf) == os.path.abspath(urdf)
        if sf and same:
            a = np.loadtxt(sf[0], delimiter=",", skiprows=1)
            series = {"t": a[:, 0], "D": a[:, 1], "HD": a[:, 5], "CD": a[:, 4]}
        self._mesh = dict(dir=d, mode=m, urdf=urdf, scene=scene, t=t, qm=qm, qr=qr, dev=dev, series=series, sel=len(t) // 2, vmax=None)
        self.mv_focus_cb["values"] = ["전체"] + scene.names
        self.mv_focus.set("전체")
        self.rfig.clear()
        self._ax_time = self.rfig.add_axes([0.085, 0.80, 0.86, 0.16])
        self._ax_mesh = self.rfig.add_axes([0.02, 0.02, 0.86, 0.70])
        self._cax = self.rfig.add_axes([0.91, 0.10, 0.022, 0.50])
        self._ax_time2 = self._cursor = None
        self._mesh_active = True
        if series is None:
            threading.Thread(target=self._mesh_series_thread, args=(d, m, urdf, t, qm, qr), daemon=True).start()
        else:
            self._mesh_after_series()
        self._mesh_draw_time()
        self._mesh_render()

    def _mesh_series_thread(self, d, m, urdf, t, qm, qr):
        """저장된 surface_*.csv 가 없으면(URDF 없이 분석한 실행) 표면 오차 시계열을 여기서 계산한다."""
        from .surface import RobotSurface, compare_poses

        try:
            robot = RobotSurface(urdf, n_points=300)
            idx = np.unique(np.linspace(0, len(t) - 1, min(len(t), 160)).astype(int))
            rows = [compare_poses(robot, qm[i], qr[i]) for i in idx]
            data = {"t": t[idx], "D": np.array([r["disp_max"] for r in rows]), "HD": np.array([r["hausdorff"] for r in rows]),
                    "CD": np.array([r["chamfer"] for r in rows])}
            self.q.put(("mesh_series", (d, m), data))
        except Exception as e:  # noqa: BLE001
            self.q.put(("log", f"표면 오차 시계열 계산 실패: {e}"))

    def _mesh_series_ready(self, key, data):
        M = self._mesh
        if not self._mesh_active or M is None or key != (M["dir"], M["mode"]):
            return
        M["series"] = data
        self._mesh_after_series()
        self._mesh_draw_time()
        self._mesh_render()

    def _mesh_after_series(self):
        """시계열이 준비되면 색 상한(시계열 최대 변위)을 정한다."""
        M = self._mesh
        M["vmax"] = float(M["series"]["D"].max()) if M["series"] is not None else None
        self._mesh_scale_label()

    def _mesh_scale_label(self):
        self.mv_val.set(f"방위 {self.mv_az.get():.0f}°  고도 {self.mv_el.get():.0f}°")

    def _mesh_view(self, az, el):
        self.mv_az.set(az); self.mv_el.set(el)
        self._mesh_scale_label()
        self._mesh_render()

    def _mesh_jump(self, where):
        M = self._mesh
        if not self._mesh_active or M is None:
            return
        if where == "first":
            M["sel"] = 0
        elif where == "last":
            M["sel"] = len(M["t"]) - 1
        elif M["series"] is not None:
            M["sel"] = int(np.argmin(np.abs(M["t"] - M["series"]["t"][int(np.argmax(M["series"]["D"]))])))
        self._mesh_update_cursor()
        self._mesh_render()

    def _mesh_draw_time(self):
        """위쪽 시간 그래프: 시점을 고르는 용도. 표면 오차(mm)와 관절 오차(deg)를 같이 보여준다."""
        M = self._mesh
        ax = self._ax_time
        if self._ax_time2 is not None:
            self._ax_time2.remove()
            self._ax_time2 = None
        ax.clear()
        ser = M["series"]
        if ser is not None:
            ax.plot(ser["t"], ser["D"], color="C3", lw=1.6, label="최악 점 변위 D")
            ax.plot(ser["t"], ser["HD"], color="C1", lw=0.9, ls="--", label="Hausdorff")
            ax.plot(ser["t"], ser["CD"], color="C0", lw=1.3, label="Chamfer")
            ax.legend(fontsize=7, loc="upper left", ncol=3)
        else:
            ax.text(0.5, 0.5, "표면 오차 계산 중…", transform=ax.transAxes, ha="center", va="center", color="#777")
        ax.set_ylabel("표면 차이 [mm]", fontsize=8)
        ax.set_xlabel("추종 시간 [s]  ← 그래프를 클릭/드래그해서 시점 선택", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(M["t"], M["dev"], color="0.55", lw=0.8)
        ax2.set_ylabel("관절 오차 d [deg]", fontsize=8, color="0.4")
        ax2.tick_params(labelsize=7, colors="0.4")
        self._ax_time2 = ax2
        self._cursor = ax.axvline(M["t"][M["sel"]], color="k", lw=1.4)
        self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
        self._mesh_update_cursor()

    def _mesh_update_cursor(self):
        M = self._mesh
        if self._cursor is not None and M is not None:
            self._cursor.set_xdata([M["t"][M["sel"]]] * 2)
            self.mv_tlabel.set(f"t = {M['t'][M['sel']]:.2f} s  (관절 오차 {M['dev'][M['sel']]:.2f}°)")
        self.rcanvas.draw_idle()

    def _mesh_mouse(self, ev):
        M = self._mesh
        if not self._mesh_active or M is None or getattr(self.rtoolbar, "mode", ""):
            return
        if ev.name == "button_release_event":
            if self._mesh_drag:
                self._mesh_drag = False
                self._mesh_render()
            return
        if ev.inaxes not in (self._ax_time, self._ax_time2) or ev.xdata is None:
            return
        if ev.name == "button_press_event":
            self._mesh_drag = True
        elif not self._mesh_drag:
            return
        M["sel"] = int(np.argmin(np.abs(M["t"] - ev.xdata)))
        self._mesh_update_cursor()

    def _mesh_render(self):
        """선택한 시점의 메시 차이를 아래쪽에 그린다. 색 = 같은 메시 점의 실제 변위, 형상은 실제 크기(×1)."""
        M = self._mesh
        if not self._mesh_active or M is None:
            return
        i = M["sel"]
        focus = self.mv_focus.get()
        vmax = M["vmax"] if (self.mv_fixed.get() and M["vmax"]) else None
        try:
            info = meshview.render(self._ax_mesh, M["scene"], M["qm"][i], M["qr"][i], azim=self.mv_az.get(), elev=self.mv_el.get(),
                                   focus=focus, vmax=vmax, show_ref=self.mv_ref.get(), show_struct=self.mv_struct.get())
        except Exception as e:  # noqa: BLE001
            self._ax_mesh.clear(); self._ax_mesh.text(0.5, 0.5, f"렌더링 실패: {e}", transform=self._ax_mesh.transAxes, ha="center")
            self.rcanvas.draw_idle()
            return
        ax = self._ax_mesh
        w_link, w_mm = info["worst"]
        ax.text(0.01, 0.99, f"t = {M['t'][i]:.2f} s   최악 점 변위 {w_mm:.1f} mm ({w_link})   관절 오차 {M['dev'][i]:.2f}°\n"
                f"색 = 같은 메시 점의 실제 변위, 형상은 실제 크기 (회색 = 기준 자세)",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.text(0.01, 0.01, "링크별 최대 변위 [mm]\n" + "\n".join(f"{n}: {v:.1f}" for n, v in info["per_link"].items()),
                transform=ax.transAxes, va="bottom", fontsize=7, color="#333")
        self._cax.clear()
        self.rfig.colorbar(ScalarMappable(norm=Normalize(0, info["vmax"]), cmap=info["cmap"]), cax=self._cax)
        self._cax.set_ylabel("변위 [mm]", fontsize=8); self._cax.tick_params(labelsize=7)
        self.rcanvas.draw_idle()

    def show_image(self):
        if not self.cur_dir or not self.img_var.get():
            return
        self._mesh_active = False
        self.mesh_frame.pack_forget()
        if self.img_var.get() == INTERACTIVE_MESH:
            self.joint_frame.pack_forget()
            self.mesh_frame.pack(fill="x", pady=2, after=self._top)
            self.show_mesh()
            return
        if self.img_var.get().startswith("★"):
            self.joint_frame.pack(fill="x", pady=2, after=self._top)
            self.show_interactive()
            return
        self.joint_frame.pack_forget()
        try:
            rebuilt = analysis.redraw_saved_graph(self.cur_dir, self.img_var.get(), self.rfig)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            rebuilt = False
            self.logmsg(f"시간축 데이터 재표시 불가, 저장 이미지 표시: {exc}")
        if rebuilt:
            self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
            self.rcanvas.draw()
            return
        img = mpimg.imread(os.path.join(self.cur_dir, self.img_var.get()))
        self.rfig.clear()
        ax = self.rfig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        self.rtime.refresh((self.cur_dir, self.cur_mode, self.img_var.get()))
        self.rcanvas.draw()

    def on_reanalyze(self):
        sel = self.tree.selection()
        if not sel:
            return
        d, _ = self.runs[sel[0]]
        try:
            s = self._read_form()
        except ValueError as e:
            messagebox.showerror("설정 오류", str(e))
            return

        def work():
            try:
                analysis.analyze(d, s, summary_dir=os.path.dirname(os.path.abspath(d)), log=self.logmsg)
                self.q.put(("done", d, None))
            except Exception as e:  # noqa: BLE001
                self.q.put(("done", d, e))
        self.status.set("분석 중…")
        threading.Thread(target=work, daemon=True).start()

    def on_open_dir(self):
        if self.cur_dir:
            open_path(self.cur_dir)

    def on_open_summary(self):
        p = os.path.join(self._out_dir(), "summary.csv")
        if os.path.exists(p):
            open_path(p)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("종료", "실행 중입니다. 중지하고 종료할까요?"):
                return
            self.stop.set()
        try:
            self._read_form().save()
        except Exception:  # noqa: BLE001
            pass
        if self.mock:
            self.mock.stop()
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    App(root)
    root.mainloop()
