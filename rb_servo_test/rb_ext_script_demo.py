"""Rainbow Robotics 외부 스크립트(External Script) API 데모.

- 명령 포트 5000: 스크립트 명령 전송 (한 줄 단위)
- 데이터 포트 5001: `reqdata` 전송 시 로봇 상태 수신

사용 예:
    python rb_ext_script_demo.py --ip 192.168.0.10 --work my_task
    python rb_ext_script_demo.py --ip 192.168.0.10 --state-only

기본값은 안전을 위해 simulation 모드이며, 실제 로봇 구동은 --real 을 명시해야 한다.
"""
import argparse
import socket
import time

CMD_PORT = 5000
DATA_PORT = 5001


class RBExtScript:
    def __init__(self, ip, timeout=3.0):
        self.cmd = socket.create_connection((ip, CMD_PORT), timeout=timeout)
        self.data = socket.create_connection((ip, DATA_PORT), timeout=timeout)

    def send(self, command, timeout=None):
        """명령 한 줄 전송 후 응답 문자열 반환. timeout: 응답 대기 시간(초)."""
        if timeout is not None:
            self.cmd.settimeout(timeout)
        self.cmd.sendall((command.strip() + "\n").encode())
        resp = self.cmd.recv(4096).decode(errors="replace").strip()
        print(f"> {command:<24} -> {resp}")
        return resp

    def request_state(self):
        """reqdata 전송 후 원본 바이트 반환 (구조는 'Request Data Structure' 문서 참고)."""
        self.data.sendall(b"reqdata\n")
        return self.data.recv(65536)

    def close(self):
        self.cmd.close()
        self.data.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="제어박스 IP (LCD에 표시됨)")
    ap.add_argument("--work", help="제어박스에 저장된 작업 파일명 (.wsl 확장자 제외)")
    ap.add_argument("--real", action="store_true", help="real 모드 (기본: simulation)")
    ap.add_argument("--init", action="store_true", help="mc jall init 실행")
    ap.add_argument("--movej", help='UI 스크립트 move_j 실행, 관절각(deg) 6개 예: "0,0,90,0,90,0"')
    ap.add_argument("--speed", type=float, default=30, help="move_j 속도 (deg/s)")
    ap.add_argument("--accel", type=float, default=50, help="move_j 가속도 (deg/s^2)")
    ap.add_argument("--state-only", action="store_true", help="상태만 조회하고 종료")
    ap.add_argument("--run-sec", type=float, default=5.0, help="실행 후 상태 모니터링 시간")
    args = ap.parse_args()

    rb = RBExtScript(args.ip)
    try:
        raw = rb.request_state()
        print(f"[state] {len(raw)} bytes received")
        if args.state_only:
            return

        if args.init:
            rb.send("mc jall init")
            time.sleep(1.0)

        rb.send(f"pgmode {'real' if args.real else 'simulation'}")

        if args.movej:
            joints = [float(v) for v in args.movej.split(",")]
            if len(joints) != 6:
                raise SystemExit("--movej 는 관절각 6개가 필요함")
            jnt = ",".join(f"{v:g}" for v in joints)
            # wait 기본값(1)이면 모션 완료 후 응답하므로 타임아웃을 넉넉히 둔다
            rb.send(f"move_j(jnt[{jnt}], {args.speed:g}, {args.accel:g})", timeout=60)
            rb.cmd.settimeout(3.0)

        if not args.work:
            print("--work 가 없어 작업 실행은 건너뜀")
            return

        rb.send(f"task load {args.work}")
        rb.send("task repeat 1")
        rb.send("task play once")

        t0 = time.time()
        while time.time() - t0 < args.run_sec:
            print(f"[state] {len(rb.request_state())} bytes")
            time.sleep(1.0)

        rb.send("task pause")
        time.sleep(0.5)
        rb.send("task resume_a")
        time.sleep(1.0)
        rb.send("task stop")
    except KeyboardInterrupt:
        rb.send("task stop")
    finally:
        rb.close()


if __name__ == "__main__":
    main()
