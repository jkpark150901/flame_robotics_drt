#!/usr/bin/env python3

import time

import numpy as np
import rbpodo as rb


ROBOT_IP = "10.0.2.7"


def _main():
    robot = rb.Cobot(ROBOT_IP)
    rc = rb.ResponseCollector()

    try:
        robot.set_operation_mode(rc, rb.OperationMode.Simulation)
        robot.set_speed_bar(rc, 1.0)
        rc.error().throw_if_not_empty()

        home = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        robot.move_j(rc, home, 50, 100)
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)
        rc.error().throw_if_not_empty()

        robot.disable_waiting_ack(rc)
        for i in range(1000):
            q = np.array([i * 90.0 / 1000.0, i * 90.0 / 1000.0, 0, 0, 0, 0], dtype=float)
            robot.move_servo_j(rc, q, 0.01, 0.1, 0.1, 0.1)
            time.sleep(0.005)

        # robot.move_speed_j(rc, q, 0.01, 0.1, 1.0, 1.0)
        robot.enable_waiting_ack(rc)
        robot.wait_for_move_finished(rc)
        rc.clear()

        robot.move_j(rc, home, 50, 100)
        if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)
        rc.error().throw_if_not_empty()

    finally:
        print("Exit")


if __name__ == "__main__":
    _main()
