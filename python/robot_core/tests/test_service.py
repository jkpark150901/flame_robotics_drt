import multiprocessing as mp
import socket
import time

from robot_core.service import (
    EmbeddedRobotCoreClient,
    ExternalRobotCoreClient,
    serve_robot_core,
)


def _serve(port):
    serve_robot_core({}, {"transport": "tcp", "channel": "127.0.0.1", "port": port})


def test_embedded_robot_core_process_starts_and_stops():
    client = EmbeddedRobotCoreClient({}, lambda _result: None)
    client.start()
    try:
        assert client.is_running
        assert client.pid is not None
    finally:
        assert client.stop(timeout=5.0)
    assert not client.is_running


def test_external_robot_core_shutdown_stops_service():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = mp.get_context("spawn").Process(target=_serve, args=(port,))
    process.start()
    from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe
    zpipe_instance = zpipe_create_pipe(io_threads=2)
    client = ExternalRobotCoreClient(
        {"transport": "tcp", "channel": "127.0.0.1", "port": port},
        lambda _result: None, zpipe=zpipe_instance)
    try:
        time.sleep(0.3)
        client.start()
        assert client.shutdown()
        process.join(5.0)
        assert not process.is_alive()
    finally:
        client.stop()
        zpipe_destroy_pipe()
        if process.is_alive():
            process.terminate()
            process.join(2.0)
