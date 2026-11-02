"""Standalone Rainbow external-script TCP terminal (standard library only)."""
import argparse
from datetime import datetime
import math
import queue
import select
import socket
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from motion import ServoProfile


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("유한한 숫자를 입력하세요.")
    return result


def build_command(kind, joints, parameters):
    angles = [number(v) for v in joints]
    if len(angles) != 6:
        raise ValueError("관절각은 6개여야 합니다.")
    values = [number(v) for v in parameters]
    if kind == "move_j":
        if len(values) != 2 or any(v <= 0 for v in values):
            raise ValueError("속도와 가속도는 0보다 커야 합니다.")
    elif kind == "move_servo_j":
        if len(values) != 4:
            raise ValueError("Servo J 파라미터는 4개여야 합니다.")
        t1, t2, gain, alpha = values
        if t1 < 0.002 or not 0.02 <= t2 <= 0.2 or not 0 < gain <= 1 or not 0 <= alpha <= 1:
            raise ValueError("t1 ≥ 0.002, 0.02 ≤ t2 ≤ 0.2, 0 < gain ≤ 1, 0 ≤ alpha ≤ 1 범위를 확인하세요.")
    else:
        raise ValueError("지원하지 않는 명령입니다.")
    return f"{kind}(jnt[{','.join(f'{v:.12g}' for v in angles)}], {', '.join(f'{v:.12g}' for v in values)})"


def encode_command(command, ending):
    if not command.strip():
        raise ValueError("전송할 명령을 입력하세요.")
    return command.encode("utf-8") + {"LF": b"\n", "CRLF": b"\r\n", "None": b""}[ending]


class Connection:
    """Owns its socket in one worker; GUI consumes timestamped events."""
    def __init__(self, host, port, events):
        self.host, self.port, self.events = host, port, events
        self.outgoing = queue.Queue(maxsize=100)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.reply = b""

    def emit(self, kind, data):
        self.events.put((datetime.now().astimezone().isoformat(timespec="milliseconds"), kind, data))

    def run(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=3) as sock:
                sock.settimeout(0.1)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.stop.is_set():
                    return
                self.emit("CONNECTED", f"{self.host}:{self.port}")
                while not self.stop.is_set():
                    try:
                        payload = self.outgoing.get_nowait()
                    except queue.Empty:
                        payload = None
                    if payload is not None:
                        if isinstance(payload, dict):
                            try:
                                self.execute_sequence(sock, payload)
                            except (ValueError, TimeoutError, InterruptedError) as exc:
                                self.emit("ERROR", str(exc))
                            finally:
                                self.emit("SEQUENCE_DONE", "웨이포인트 전송 종료 (실측 도달 여부는 확인하지 않음)")
                        else:
                            self.transmit(sock, payload)
                    self.receive(sock, 0.005)
        except EOFError as exc:
            self.emit("INFO", str(exc))
        except OSError as exc:
            self.emit("ERROR", str(exc))
        finally:
            self.emit("DISCONNECTED", "연결 해제 (로봇 동작 정지를 의미하지 않습니다.)")

    def transmit(self, sock, payload):
        sock.sendall(payload)
        self.emit("TX", payload)

    def receive(self, sock, timeout):
        if select.select([sock], [], [], timeout)[0]:
            data = sock.recv(65536)
            if not data:
                raise EOFError("상대 장치가 연결을 종료했습니다.")
            self.emit("RX", data)
            self.reply = (self.reply + data)[-16384:]

    def check_sequence(self, plan):
        if self.stop.is_set() or plan["cancel"].is_set():
            raise InterruptedError("웨이포인트 전송을 중단했습니다. 이미 전송된 로봇 동작은 자동 정지되지 않습니다.")
        reply = self.reply.lower()
        if b"not allowed" in reply or b"error" in reply:
            raise ValueError("로봇이 오류/거부 응답을 반환했습니다. RX 로그를 확인하세요.")

    def wait_until(self, sock, deadline, plan):
        while True:
            self.check_sequence(plan)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self.receive(sock, min(remaining, 0.005))

    def move_and_wait(self, sock, point, plan):
        self.check_sequence(plan)
        self.reply = b""
        # Explicit wait=1: the script returns after this move completes.
        command = build_command("move_j", point, [plan["speed"], plan["accel"]])
        command = command[:-1] + ", 1)"
        self.transmit(sock, encode_command(command, plan["ending"]))
        deadline = time.perf_counter() + plan["timeout"]
        while b"The command was executed" not in self.reply:
            self.check_sequence(plan)
            if time.perf_counter() >= deadline:
                raise TimeoutError("Move J 완료 응답 대기 시간 초과. 후속 명령 전송을 중단합니다.")
            self.receive(sock, 0.005)
        self.check_sequence(plan)

    def execute_sequence(self, sock, plan):
        self.reply = b""
        self.check_sequence(plan)
        if plan["kind"] == "move_j":
            for i, point in enumerate(plan["points"], 1):
                self.move_and_wait(sock, point, plan)
                self.emit("PROGRESS", f"Move J {i}/{len(plan['points'])}")
            return
        profile = plan["profile"]
        self.emit("PROGRESS", "첫 웨이포인트로 Move J 이동 / 완료 응답 대기")
        self.move_and_wait(sock, profile.points[0], plan)
        self.reply = b""
        start = time.perf_counter()
        period = profile.period
        self.emit("INFO", f"Servo 프로파일 {profile.duration:.3f}s / {profile.count}개 지령 / 각 웨이포인트에서 감속 정지")
        for i, (offset, point) in enumerate(profile.samples(), 1):
            self.wait_until(sock, start + offset, plan)
            lag = time.perf_counter() - start - offset
            if lag > max(0.05, 2 * period):
                raise TimeoutError(f"Servo 전송 지연 {lag * 1000:.1f}ms. 밀린 지령을 몰아서 보내지 않고 중단합니다.")
            command = build_command("move_servo_j", point, [period, plan["t2"], plan["gain"], plan["alpha"]])
            self.transmit(sock, encode_command(command, plan["ending"]))
            if i % max(1, round(0.2 / period)) == 0 or i == profile.count:
                self.emit("PROGRESS", f"Servo J {i}/{profile.count} · {offset:.2f}/{profile.duration:.2f}s")
        # Repeat the endpoint to remove the final target velocity (same as runner.py).
        for hold in range(2):
            self.wait_until(sock, start + profile.duration + hold * period, plan)
            command = build_command("move_servo_j", profile.points[-1], [period, 0.02, plan["gain"], plan["alpha"]])
            self.transmit(sock, encode_command(command, plan["ending"]))
        self.wait_until(sock, start + profile.duration + 2 * period + plan["t2"], plan)


class App(tk.Tk):
    def __init__(self, host="192.168.0.10", port=5000):
        super().__init__()
        self.title("Rainbow Robotics · External Script TCP Test")
        self.geometry("1350x850")
        self.minsize(1100, 760)
        self.events = queue.Queue()
        self.connection = None
        self.connected = False
        self.sequence_cancel = None
        self.protocol("WM_DELETE_WINDOW", self.close)
        bar = ttk.Frame(self, padding=10)
        bar.pack(fill="x")
        self.host = tk.StringVar(value=host)
        self.port = tk.StringVar(value=str(port))
        ttk.Label(bar, text="IP / Host").pack(side="left")
        self.host_entry = ttk.Entry(bar, textvariable=self.host, width=22)
        self.host_entry.pack(side="left", padx=6)
        ttk.Label(bar, text="Port").pack(side="left")
        self.port_entry = ttk.Entry(bar, textvariable=self.port, width=7)
        self.port_entry.pack(side="left", padx=6)
        self.connect_button = ttk.Button(bar, text="연결", command=self.connect)
        self.connect_button.pack(side="left", padx=6)
        self.disconnect_button = ttk.Button(bar, text="연결 해제", command=self.disconnect, state="disabled")
        self.disconnect_button.pack(side="left")
        self.status = tk.StringVar(value="연결 안 됨")
        ttk.Label(bar, textvariable=self.status).pack(side="right")
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left, right = ttk.Frame(panes, padding=8), ttk.Frame(panes, padding=8)
        panes.add(left, weight=2)
        panes.add(right, weight=3)
        tabs = ttk.Notebook(left)
        tabs.pack(fill="both", expand=True)
        motion_tab, terminal_tab = ttk.Frame(tabs, padding=5), ttk.Frame(tabs, padding=5)
        tabs.add(motion_tab, text="관절 / 웨이포인트")
        tabs.add(terminal_tab, text="직접 스크립트")
        left = motion_tab
        joints_box = ttk.LabelFrame(left, text="목표 관절각 [deg]", padding=8)
        joints_box.pack(fill="x")
        self.joints = []
        for i in range(6):
            ttk.Label(joints_box, text=f"J{i + 1}").grid(row=i // 3 * 2, column=i % 3, sticky="w")
            value = tk.StringVar(value="0")
            ttk.Entry(joints_box, textvariable=value, width=12).grid(row=i // 3 * 2 + 1, column=i % 3, padx=4, pady=4)
            self.joints.append(value)
        point_buttons = ttk.Frame(left)
        point_buttons.pack(fill="x", pady=4)
        for label, callback in [("현재 입력 추가", self.add_waypoint), ("선택 행 수정", self.update_waypoint), ("선택 삭제", self.delete_waypoint), ("↑", lambda: self.shift_waypoint(-1)), ("↓", lambda: self.shift_waypoint(1))]:
            ttk.Button(point_buttons, text=label, command=callback).pack(side="left", padx=1)
        table_box = ttk.Frame(left)
        table_box.pack(fill="both", expand=True)
        self.waypoints = ttk.Treeview(table_box, columns=tuple(f"J{i}" for i in range(1, 7)), show="headings", height=6, selectmode="browse")
        for col in self.waypoints["columns"]:
            self.waypoints.heading(col, text=col)
            self.waypoints.column(col, width=65, minwidth=40, anchor="e", stretch=True)
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self.waypoints.yview)
        self.waypoints.configure(yscrollcommand=scroll.set)
        self.waypoints.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.waypoints.bind("<<TreeviewSelect>>", self.load_waypoint)
        ttk.Label(left, text="단일 전송: 위 입력값 / 전체 실행: 표의 위→아래 순서").pack(anchor="w")
        self.motion_variables = {}
        self.motion_panel(left, "Move J", "move_j", [("Speed [deg/s]", "10"), ("Accel [deg/s²]", "20")])
        self.motion_panel(left, "Servo J · move_servo_j", "move_servo_j", [("t1 / 주기 [s]", "0.02"), ("t2 [s]", "0.04"), ("Gain", "0.1"), ("Alpha", "0.5"), ("Speed [deg/s]", "10"), ("Accel [deg/s²]", "20")])
        run_bar = ttk.Frame(left)
        run_bar.pack(fill="x", pady=4)
        self.cancel_button = ttk.Button(run_bar, text="웨이포인트 전송 중단", command=self.cancel_sequence, state="disabled")
        self.cancel_button.pack(side="left")
        ttk.Label(run_bar, text="Move J 응답 제한 [s]").pack(side="left", padx=5)
        self.move_timeout = tk.StringVar(value="120")
        ttk.Entry(run_bar, textvariable=self.move_timeout, width=7).pack(side="left")
        self.sequence_status = tk.StringVar(value="실행 대기")
        ttk.Label(left, textvariable=self.sequence_status).pack(anchor="w")
        custom = ttk.LabelFrame(terminal_tab, text="직접 스크립트 입력 (UTF-8)", padding=8)
        custom.pack(fill="both", expand=True, pady=6)
        self.script = scrolledtext.ScrolledText(custom, height=4, width=42, font="TkFixedFont")
        self.script.pack(fill="both", expand=True)
        ttk.Button(custom, text="입력 전문 전송", command=lambda: self.send(self.script.get("1.0", "end-1c"))).pack(anchor="e", pady=4)
        options = ttk.Frame(self)
        options.pack(fill="x")
        ttk.Label(options, text="전송 끝 문자").pack(side="left")
        self.ending = tk.StringVar(value="LF")
        ttk.Combobox(options, textvariable=self.ending, values=("LF", "CRLF", "None"), state="readonly", width=7).pack(side="left", padx=5)
        modes = ttk.Frame(self)
        modes.pack(fill="x", pady=6)
        ttk.Button(modes, text="Task Stop", command=lambda: self.send("task stop")).pack(side="left", padx=2)
        ttk.Label(motion_tab, text="Servo 전체 실행: 첫 점 Move J 이동 후 속도·가속도 보간 추종\n전송 중단/연결 해제는 로봇 정지가 아닙니다. Task Stop은 비상정지가 아닙니다.", foreground="#805000").pack(anchor="w")
        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="RAW 송수신 로그 · 바이트 원문 / HEX").pack(side="left")
        ttk.Button(toolbar, text="지우기", command=self.clear_log).pack(side="right")
        ttk.Button(toolbar, text="저장", command=self.save_log).pack(side="right", padx=4)
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="자동 스크롤", variable=self.autoscroll).pack(side="right")
        self.log = scrolledtext.ScrolledText(right, state="disabled", wrap="word", font="TkFixedFont", background="#121923", foreground="#d9e2ef", width=65)
        self.log.pack(fill="both", expand=True, pady=6)
        for tag, color in [("TX", "#79cefa"), ("RX", "#9de0ad"), ("ERROR", "#ff9898")]:
            self.log.tag_configure(tag, foreground=color)
        self.after(50, self.poll)

    def motion_panel(self, parent, title, kind, fields):
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(fill="x", pady=6)
        variables = []
        for i, (label, default) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=i // 2, column=i % 2 * 2, sticky="w", padx=3)
            value = tk.StringVar(value=default)
            ttk.Entry(box, textvariable=value, width=9).grid(row=i // 2, column=i % 2 * 2 + 1, pady=3)
            variables.append(value)
        self.motion_variables[kind] = variables
        row = (len(fields) + 1) // 2
        ttk.Button(box, text="입력값 단일 전송", command=lambda: self.motion(kind, variables)).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(box, text="웨이포인트 전체 실행", command=lambda: self.start_sequence(kind)).grid(row=row, column=2, columnspan=2, sticky="ew", pady=4)

    def motion(self, kind, variables):
        try:
            command = build_command(kind, self.input_point(), [v.get() for v in variables[:4 if kind == "move_servo_j" else 2]])
        except ValueError as exc:
            messagebox.showerror("입력 오류", str(exc))
            return
        self.send(command)

    def input_point(self):
        return [number(v.get()) for v in self.joints]

    def add_waypoint(self):
        try:
            self.waypoints.insert("", "end", values=self.input_point())
        except ValueError as exc:
            messagebox.showerror("입력 오류", str(exc))

    def update_waypoint(self):
        selected = self.waypoints.selection()
        if selected:
            try:
                self.waypoints.item(selected[0], values=self.input_point())
            except ValueError as exc:
                messagebox.showerror("입력 오류", str(exc))

    def delete_waypoint(self):
        for item in self.waypoints.selection():
            self.waypoints.delete(item)

    def shift_waypoint(self, direction):
        selected = self.waypoints.selection()
        if selected:
            item = selected[0]
            index = max(0, min(len(self.waypoints.get_children()) - 1, self.waypoints.index(item) + direction))
            self.waypoints.move(item, "", index)

    def load_waypoint(self, _event=None):
        selected = self.waypoints.selection()
        if selected:
            for variable, value in zip(self.joints, self.waypoints.item(selected[0], "values")):
                variable.set(value)

    def start_sequence(self, kind):
        if not self.connected or self.sequence_cancel is not None:
            messagebox.showerror("실행 불가", "TCP 연결을 확인하고 기존 실행이 종료된 뒤 시작하세요.")
            return
        try:
            if self.ending.get() == "None":
                raise ValueError("웨이포인트 실행은 LF 또는 CRLF 끝 문자가 필요합니다.")
            points = [[number(v) for v in self.waypoints.item(item, "values")] for item in self.waypoints.get_children()]
            if not points:
                raise ValueError("웨이포인트를 먼저 추가하세요.")
            values = [number(v.get()) for v in self.motion_variables[kind]]
            speed, accel = values[-2:]
            build_command("move_j", points[0], [speed, accel])
            timeout = number(self.move_timeout.get())
            if timeout <= 0:
                raise ValueError("응답 제한 시간은 0보다 커야 합니다.")
            plan = dict(kind=kind, points=points, speed=speed, accel=accel, timeout=timeout,
                        ending=self.ending.get(), cancel=threading.Event())
            if kind == "move_servo_j":
                period, t2, gain, alpha = values[:4]
                build_command(kind, points[0], [period, t2, gain, alpha])
                if t2 < period:
                    raise ValueError("Servo 전체 실행에서 t2는 전송 주기 이상이어야 합니다.")
                plan.update(profile=ServoProfile(points, speed, accel, period), t2=t2, gain=gain, alpha=alpha)
            self.connection.outgoing.put_nowait(plan)
            self.sequence_cancel = plan["cancel"]
            self.cancel_button.configure(state="normal")
            self.sequence_status.set("웨이포인트 실행 중 (시작 시 설정값 적용)")
        except (ValueError, queue.Full) as exc:
            messagebox.showerror("실행 오류", str(exc) or "전송 대기열이 가득 찼습니다.")

    def cancel_sequence(self):
        if self.sequence_cancel is not None:
            self.sequence_cancel.set()
            self.sequence_status.set("전송 중단 요청 중…")

    def connect(self):
        try:
            port = int(self.port.get())
            if not 1 <= port <= 65535 or not self.host.get().strip():
                raise ValueError()
        except ValueError:
            messagebox.showerror("입력 오류", "IP/Host와 포트(1~65535)를 확인하세요.")
            return
        self.connection = Connection(self.host.get().strip(), port, self.events)
        self.connect_button.configure(state="disabled")
        self.host_entry.configure(state="disabled")
        self.port_entry.configure(state="disabled")
        self.disconnect_button.configure(state="normal")
        self.status.set("연결 중…")
        self.connection.thread.start()

    def disconnect(self):
        self.cancel_sequence()
        if self.connection:
            self.connection.stop.set()
            self.connected = False
            self.status.set("연결 해제 중…")
            self.disconnect_button.configure(state="disabled")

    def send(self, command):
        if not self.connected:
            messagebox.showerror("연결 필요", "먼저 TCP 연결을 완료하세요.")
            return
        if self.sequence_cancel is not None:
            if command == "task stop":
                self.cancel_sequence()
            else:
                messagebox.showerror("실행 중", "웨이포인트 전송 종료 후 수동 명령을 보내세요.")
                return
        try:
            self.connection.outgoing.put_nowait(encode_command(command, self.ending.get()))
        except (ValueError, queue.Full) as exc:
            messagebox.showerror("전송 오류", str(exc) or "전송 대기열이 가득 찼습니다.")

    def poll(self):
        for _ in range(200):
            try:
                stamp, kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "CONNECTED":
                self.connected = not self.connection.stop.is_set()
                self.status.set("연결됨" if self.connected else "연결 해제 중…")
            elif kind == "DISCONNECTED":
                self.finish_sequence()
                self.connected = False
                self.connection = None
                self.status.set("연결 안 됨")
                for widget in (self.connect_button, self.host_entry, self.port_entry):
                    widget.configure(state="normal")
                self.disconnect_button.configure(state="disabled")
            elif kind == "SEQUENCE_DONE":
                self.finish_sequence()
            elif kind == "PROGRESS":
                self.sequence_status.set(data)
            if isinstance(data, bytes):
                detail = f"{len(data)} bytes  {data!r}\n    HEX: {data.hex(' ')}"
            else:
                detail = data
            self.log.configure(state="normal")
            self.log.insert("end", f"[{stamp}] {kind}  {detail}\n", kind)
            # Keep a bounded live view; save before clearing for longer captures.
            if int(self.log.index("end-1c").split('.')[0]) > 20000:
                self.log.delete("1.0", "1001.0")
            self.log.configure(state="disabled")
            if self.autoscroll.get():
                self.log.see("end")
        self.after(50, self.poll)

    def finish_sequence(self):
        self.sequence_cancel = None
        self.cancel_button.configure(state="disabled")
        self.sequence_status.set("전송 종료 · 결과는 RX/ERROR 로그 확인")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def save_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".log", filetypes=[("Log", "*.log"), ("All", "*.*")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as output:
                    output.write(self.log.get("1.0", "end-1c"))
            except OSError as exc:
                messagebox.showerror("저장 오류", str(exc))

    def close(self):
        self.disconnect()
        self.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.0.10")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    App(args.ip, args.port).mainloop()
