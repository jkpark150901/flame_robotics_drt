"""
check_mocap_unicast.py
======================
NatNet 유니캐스트 모션캡처 데이터 수신 테스트.

precision_eval.py 와 같은 NatNetClient 콜백 방식으로 연결하고,
지정한 rigid body / labeled marker 데이터가 실제로 들어오는지 확인합니다.

사용 예:
  python check_mocap_unicast.py --model_id 1 --marker_id 0 --rigid_body_id 1 \
      --server 10.0.2.10 --client 10.0.2.20

기본값은 유니캐스트입니다. 멀티캐스트 테스트가 필요할 때만 --multicast 를 사용하세요.
"""

import argparse
import csv
import ipaddress
import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools.NatNet.NatNetClient import NatNetClient


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


@dataclass
class RigidBodySample:
    timestamp: float
    rb_id: int
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass
class LabeledMarkerSample:
    timestamp: float
    model_id: int
    marker_id: int
    position: tuple[float, float, float]


class MocapReceiveState:
    def __init__(self, model_id: int, marker_id: int, rigid_body_id: int):
        self.model_id = model_id
        self.marker_id = marker_id
        self.rigid_body_id = rigid_body_id
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.frame_count = 0
        self.filtered_rb_count = 0
        self.filtered_lm_count = 0
        self.last_frame_time: float | None = None
        self.last_rb: RigidBodySample | None = None
        self.last_lm: LabeledMarkerSample | None = None
        self.records: list[dict] = []

    def on_rigid_body(self, rb_id, position, rotation):
        if rb_id != self.rigid_body_id:
            return

        now = time.time()
        sample = RigidBodySample(
            timestamp=now,
            rb_id=int(rb_id),
            position=tuple(float(v) for v in position),
            rotation_xyzw=tuple(float(v) for v in rotation),
        )

        with self._lock:
            self.filtered_rb_count += 1
            self.last_rb = sample
            self.records.append(
                {
                    "elapsed_s": round(now - self.start_time, 6),
                    "type": "rigid_body",
                    "id": sample.rb_id,
                    "model_id": "",
                    "marker_id": "",
                    "x_m": sample.position[0],
                    "y_m": sample.position[1],
                    "z_m": sample.position[2],
                    "qx": sample.rotation_xyzw[0],
                    "qy": sample.rotation_xyzw[1],
                    "qz": sample.rotation_xyzw[2],
                    "qw": sample.rotation_xyzw[3],
                }
            )

    def on_frame(self, data_dict):
        now = time.time()
        with self._lock:
            self.frame_count += 1
            self.last_frame_time = now

        mocap_data = data_dict.get("mocap_data")
        if mocap_data is None:
            return

        lm_data = getattr(mocap_data, "labeled_marker_data", None)
        if lm_data is None:
            return

        for lm in lm_data.labeled_marker_list:
            model_id = lm.id_num >> 16
            marker_id = lm.id_num & 0xFFFF
            if model_id != self.model_id or marker_id != self.marker_id:
                continue

            sample = LabeledMarkerSample(
                timestamp=now,
                model_id=int(model_id),
                marker_id=int(marker_id),
                position=tuple(float(v) for v in lm.pos),
            )

            with self._lock:
                self.filtered_lm_count += 1
                self.last_lm = sample
                self.records.append(
                    {
                        "elapsed_s": round(now - self.start_time, 6),
                        "type": "labeled_marker",
                        "id": "",
                        "model_id": sample.model_id,
                        "marker_id": sample.marker_id,
                        "x_m": sample.position[0],
                        "y_m": sample.position[1],
                        "z_m": sample.position[2],
                        "qx": "",
                        "qy": "",
                        "qz": "",
                        "qw": "",
                    }
                )

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(time.time() - self.start_time, 1e-9)
            return {
                "elapsed": elapsed,
                "frame_count": self.frame_count,
                "frame_hz": self.frame_count / elapsed,
                "filtered_rb_count": self.filtered_rb_count,
                "filtered_rb_hz": self.filtered_rb_count / elapsed,
                "filtered_lm_count": self.filtered_lm_count,
                "filtered_lm_hz": self.filtered_lm_count / elapsed,
                "last_frame_age": None
                if self.last_frame_time is None
                else time.time() - self.last_frame_time,
                "last_rb": self.last_rb,
                "last_lm": self.last_lm,
                "records": list(self.records),
            }


def _format_pos(position: tuple[float, float, float] | None) -> str:
    if position is None:
        return "N/A"
    return f"[{position[0]: .4f}, {position[1]: .4f}, {position[2]: .4f}] m"


def _print_status(state: MocapReceiveState):
    snap = state.snapshot()
    last_rb = snap["last_rb"]
    last_lm = snap["last_lm"]

    rb_pos = _format_pos(last_rb.position if last_rb else None)
    lm_pos = _format_pos(last_lm.position if last_lm else None)
    frame_age = snap["last_frame_age"]
    frame_age_text = "N/A" if frame_age is None else f"{frame_age:.3f}s"

    log.info(
        "frames=%d (%.1f Hz, last age=%s), rb[%d]=%d (%.1f Hz) %s, "
        "marker[%d:%d]=%d (%.1f Hz) %s",
        snap["frame_count"],
        snap["frame_hz"],
        frame_age_text,
        state.rigid_body_id,
        snap["filtered_rb_count"],
        snap["filtered_rb_hz"],
        rb_pos,
        state.model_id,
        state.marker_id,
        snap["filtered_lm_count"],
        snap["filtered_lm_hz"],
        lm_pos,
    )


def _save_csv(records: list[dict], path: Path):
    if not records:
        log.warning("CSV 저장 생략: 기록된 샘플이 없습니다.")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    log.info("CSV 저장 완료: %s (%d samples)", path, len(records))


def _resolve_client_ip(server_ip: str, client_ip: str | None) -> str:
    if client_ip and client_ip.lower() != "auto":
        return client_ip

    server_addr = socket.gethostbyname(server_ip)
    if ipaddress.ip_address(server_addr).is_loopback:
        return "127.0.0.1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect does not send packets; it only lets the OS choose the
        # outbound interface that would reach the NatNet server.
        sock.connect((server_addr, 1510))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _validate_ip_pair(server_ip: str, client_ip: str):
    server_addr = ipaddress.ip_address(socket.gethostbyname(server_ip))
    client_addr = ipaddress.ip_address(socket.gethostbyname(client_ip))

    if server_addr.is_loopback != client_addr.is_loopback:
        raise ValueError(
            f"server={server_ip}, client={client_ip} 조합이 맞지 않습니다. "
            "원격 Motive 서버를 쓸 때 client는 이 PC의 같은 네트워크 대역 IP여야 합니다."
        )


def run(args) -> int:
    rigid_body_id = args.rigid_body_id
    if rigid_body_id is None:
        rigid_body_id = args.model_id

    client_ip = _resolve_client_ip(args.server, args.client)
    _validate_ip_pair(args.server, client_ip)

    state = MocapReceiveState(
        model_id=args.model_id,
        marker_id=args.marker_id,
        rigid_body_id=rigid_body_id,
    )

    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(args.server)
    client.new_frame_with_data_listener = state.on_frame
    client.rigid_body_listener = state.on_rigid_body
    client.set_use_multicast(args.multicast)
    if args.force_version:
        major, minor = args.force_version
        # version negotiation may fail over unicast; force parser to correct version
        client._NatNetClient__nat_net_requested_version[0] = major
        client._NatNetClient__nat_net_requested_version[1] = minor
        log.info("NatNet 버전 강제 설정: %d.%d", major, minor)

    mode = "multicast" if args.multicast else "unicast"
    log.info(
        "NatNet %s 수신 테스트 시작: client=%s, server=%s, rb_id=%d, marker=%d:%d",
        mode,
        client_ip,
        args.server,
        rigid_body_id,
        args.model_id,
        args.marker_id,
    )

    try:
        if not client.run("d"):
            raise RuntimeError("NatNet 스트리밍 시작 실패.")

        log.info("command socket bound: %s", client.command_socket.getsockname())
        log.info("data socket bound: %s", client.data_socket.getsockname())

        connect_deadline = time.time() + args.connect_timeout
        while time.time() < connect_deadline:
            if client.connected():
                break
            if state.snapshot()["frame_count"] > 0:
                break
            time.sleep(0.1)

        if client.connected():
            log.info("NatNet command 연결 완료.")
        else:
            log.warning(
                "NatNet command 응답은 아직 없습니다. data port 수신은 계속 확인합니다. "
                "1510 응답 경로 또는 Motive command 설정을 확인하세요."
            )

        log.info("%.1f초 동안 수신 상태를 확인합니다.", args.duration)

        end_time = time.time() + args.duration
        next_report = time.time()
        while time.time() < end_time:
            now = time.time()
            if now >= next_report:
                _print_status(state)
                next_report = now + args.report_interval
            time.sleep(0.05)

        _print_status(state)
        snap = state.snapshot()

        if args.csv:
            _save_csv(snap["records"], Path(args.csv))

        rb_ok = args.skip_rigid_body or snap["filtered_rb_count"] > 0
        lm_ok = args.skip_marker or snap["filtered_lm_count"] > 0
        frame_ok = snap["frame_count"] > 0

        if frame_ok and rb_ok and lm_ok:
            log.info("수신 테스트 성공.")
            return 0

        log.error(
            "수신 테스트 실패: frame_ok=%s, rigid_body_ok=%s, marker_ok=%s",
            frame_ok,
            rb_ok,
            lm_ok,
        )
        return 2
    finally:
        if client.command_socket is not None and client.data_socket is not None:
            client.shutdown()


def main():
    p = argparse.ArgumentParser(
        description="NatNet 유니캐스트 모션캡처 데이터 수신 테스트"
    )
    p.add_argument("--model_id", type=int, required=True, help="NatNet labeled marker model ID")
    p.add_argument("--marker_id", type=int, default=0, help="NatNet labeled marker ID")
    p.add_argument(
        "--rigid_body_id",
        type=int,
        default=None,
        help="NatNet rigid body ID (기본값: model_id 와 동일)",
    )
    p.add_argument("--server", default="10", help="NatNet 서버 IP")
    p.add_argument(
        "--client",
        default="auto",
        help="NatNet 클라이언트 IP. 기본값 auto는 server로 가는 로컬 NIC IP 자동 선택",
    )
    p.add_argument("--duration", type=float, default=10.0, help="수신 테스트 시간, 초")
    p.add_argument("--connect_timeout", type=float, default=5.0, help="서버 연결 대기 시간, 초")
    p.add_argument("--report_interval", type=float, default=1.0, help="상태 출력 주기, 초")
    p.add_argument("--csv", default=None, help="수신 샘플을 CSV로 저장할 경로")
    p.add_argument("--skip_rigid_body", action="store_true", help="rigid body 수신 여부를 성공 조건에서 제외")
    p.add_argument("--skip_marker", action="store_true", help="labeled marker 수신 여부를 성공 조건에서 제외")
    p.add_argument("--multicast", action="store_true", default=False, help="멀티캐스트 사용")
    p.add_argument(
        "--force_version",
        type=int,
        nargs=2,
        metavar=("MAJOR", "MINOR"),
        default=None,
        help="NatNet 버전 강제 설정 (예: --force_version 4 2). "
             "command port 응답이 없을 때 파싱 버전을 직접 지정.",
    )

    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
