import queue
import socket
import threading
import time
import unittest

from app import Connection, build_command, encode_command
from motion import ServoProfile


class ProtocolTest(unittest.TestCase):
    def test_profile_limits_at_reversals_and_short_segments(self):
        points = [[0] * 6, [10, -3, 2, 0, 0, 0], [-2, 8, 0, 0, 0, 0],
                  [-2, 8, 0, 0, 0, 0], [-2.0001, 8, 0, 0, 0, 0]]
        profile = ServoProfile(points, 7, 13, .02)
        samples = list(profile.samples())
        self.assertEqual(len(samples), profile.count)
        self.assertEqual(samples[-1][1], tuple(points[-1]))
        positions = [points[0]] + [q for _, q in samples] + [points[-1]]
        velocities = [[(b - a) / profile.period for a, b in zip(p, q)]
                      for p, q in zip(positions, positions[1:])]
        velocities.insert(0, [0] * 6)
        self.assertLessEqual(max(abs(v) for vs in velocities for v in vs), 7 + 1e-8)
        accelerations = [(b - a) / profile.period for p, q in zip(velocities, velocities[1:]) for a, b in zip(p, q)]
        self.assertLessEqual(max(map(abs, accelerations)), 13 + 1e-8)
        for point in points:
            self.assertIn(tuple(point), [tuple(p) for p in positions])
        for i, (offset, _) in enumerate(samples):
            self.assertAlmostEqual(offset, i * profile.period)

    def test_profile_validation(self):
        for points, speed, accel, period in [([[0] * 6] * 2, 10, 20, .02),
                                             ([[0] * 6, [1] * 6], 0, 20, .02),
                                             ([[0] * 6, [1] * 6], 10, 20, .001),
                                             ([[0] * 6, [float('inf')] * 6], 10, 20, .02)]:
            with self.assertRaises(ValueError):
                ServoProfile(points, speed, accel, period)

    def run_sequence(self, plan, responder):
        events, received = queue.Queue(), queue.Queue()
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(4)

            def serve():
                with server.accept()[0] as peer:
                    peer.settimeout(4)
                    pending = b""
                    while True:
                        chunk = peer.recv(4096)
                        if not chunk:
                            break
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            received.put((time.perf_counter(), line))
                            response = responder(line)
                            if response:
                                peer.sendall(response)

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            client = Connection("127.0.0.1", server.getsockname()[1], events)
            client.thread.start()
            errors = []
            try:
                self.assertEqual(events.get(timeout=4)[1], "CONNECTED")
                client.outgoing.put(plan)
                while True:
                    _, kind, data = events.get(timeout=4)
                    if kind == "ERROR":
                        errors.append(data)
                    if kind == "SEQUENCE_DONE":
                        break
            finally:
                client.stop.set()
                client.thread.join(timeout=4)
                worker.join(timeout=4)
            lines = []
            while not received.empty():
                lines.append(received.get_nowait())
            return lines, errors

    def sequence_plan(self, kind="move_servo_j"):
        points = [[0] * 6, [.1, 0, 0, 0, 0, 0], [0] * 6]
        return dict(kind=kind, points=points, profile=ServoProfile(points, 10, 100, .02),
                    speed=10, accel=100, t2=.04, gain=.1, alpha=.5,
                    ending="LF", timeout=.2, cancel=threading.Event())

    def test_servo_stream_does_not_wait_for_each_reply(self):
        plan = self.sequence_plan()
        lines, errors = self.run_sequence(plan, lambda line: b"The command was executed\n" if line.startswith(b"move_j(") else None)
        self.assertEqual(errors, [])
        self.assertTrue(lines[0][1].endswith(b", 1)"))
        self.assertEqual(len(lines), 1 + plan["profile"].count + 2)
        self.assertTrue(all(line.startswith(b"move_servo_j(") for _, line in lines[1:]))
        self.assertGreater(lines[-1][0] - lines[1][0], plan["profile"].duration * .8)

    def test_sequence_waits_and_aborts_on_timeout_or_denial(self):
        for responder in [lambda line: None, lambda line: b"The command is not allowed\n"]:
            lines, errors = self.run_sequence(self.sequence_plan(), responder)
            self.assertEqual(len(lines), 1)
            self.assertEqual(len(errors), 1)

    def test_move_j_sequence_and_cancel(self):
        plan = self.sequence_plan("move_j")
        lines, errors = self.run_sequence(plan, lambda line: b"The command was executed\n")
        self.assertEqual(errors, [])
        self.assertEqual(len(lines), 3)
        plan = self.sequence_plan()

        def cancel_on_first(line):
            plan["cancel"].set()
            return b"The command was executed\n"

        lines, errors = self.run_sequence(plan, cancel_on_first)
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(errors), 1)

    def test_motion_commands(self):
        self.assertEqual(build_command("move_j", [0, -10, 90, 0, 90, 0], [10, 20]),
                         "move_j(jnt[0,-10,90,0,90,0], 10, 20)")
        self.assertEqual(build_command("move_servo_j", [0] * 6, [.02, .04, .1, .5]),
                         "move_servo_j(jnt[0,0,0,0,0,0], 0.02, 0.04, 0.1, 0.5)")
        for joints, params in [([0] * 5, [10, 20]), ([float('nan')] * 6, [10, 20]),
                               ([0] * 6, [0, 20])]:
            with self.assertRaises(ValueError):
                build_command("move_j", joints, params)
        with self.assertRaises(ValueError):
            build_command("move_servo_j", [0] * 6, [.02, .5, .1, .5])

    def test_line_endings(self):
        for ending, suffix in [("LF", b"\n"), ("CRLF", b"\r\n"), ("None", b"")]:
            self.assertEqual(encode_command(" cmd ", ending), b" cmd " + suffix)

    def test_tcp_raw_and_remote_close(self):
        events = queue.Queue()
        received = queue.Queue()
        payload = b"move_j(jnt[0,0,0,0,0,0], 10, 20)\n"
        response = b"OK\r\n\x00\xff"
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(4)

            def serve():
                with server.accept()[0] as peer:
                    peer.settimeout(4)
                    data = b""
                    while not data.endswith(b"\n"):
                        chunk = peer.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    received.put(data)
                    peer.sendall(response[:2])
                    peer.sendall(response[2:])

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            client = Connection("127.0.0.1", server.getsockname()[1], events)
            client.thread.start()
            try:
                self.assertEqual(events.get(timeout=4)[1], "CONNECTED")
                client.outgoing.put(payload)
                tx, rx = [], []
                while True:
                    _, kind, data = events.get(timeout=4)
                    if kind == "TX":
                        tx.append(data)
                    elif kind == "RX":
                        rx.append(data)
                    elif kind == "ERROR":
                        self.fail(data)
                    elif kind == "DISCONNECTED":
                        break
                self.assertEqual(received.get(timeout=4), payload)
                self.assertEqual(tx, [payload])
                self.assertEqual(b"".join(rx), response)
            finally:
                client.stop.set()
                client.thread.join(timeout=4)
                worker.join(timeout=4)
            self.assertFalse(client.thread.is_alive())

    def test_disconnect_while_idle(self):
        events = queue.Queue()
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(4)
            client = Connection("127.0.0.1", server.getsockname()[1], events)
            client.thread.start()
            with server.accept()[0]:
                self.assertEqual(events.get(timeout=4)[1], "CONNECTED")
                client.stop.set()
                client.thread.join(timeout=4)
                self.assertFalse(client.thread.is_alive())
                self.assertEqual(events.get(timeout=4)[1], "DISCONNECTED")


if __name__ == "__main__":
    unittest.main()
