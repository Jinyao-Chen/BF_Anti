#!/usr/bin/env python3
"""RotorForce bridge: motor commands -> body wrench for a force-only plant.

This is the first step of the long-term ``RotorForcePlugin`` path.  Instead of
spinning rotor links and running Gazebo's MulticopterMotorModel (which triggers
the ODE collision assertion), it computes the total thrust / body torque from
the four normalized motor commands and publishes a ``gz.msgs.EntityWrench`` to
the ``gz-sim-apply-link-wrench-system`` plugin.

Two modes:
  * ``--mode telemetry``: subscribe to ``/betaflight/telemetry`` (Float_V,
    data[0..3] = motor commands, data[4] = world z, data[5..7] = roll/pitch/yaw
    in rad), and apply the computed wrench.
  * ``--mode hover``: publish a constant weight-compensating wrench.  Useful for
    a standalone smoke test of the force world without Betaflight.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List

import gz.transport13 as transport
from gz.msgs10 import entity_pb2
from gz.msgs10 import entity_wrench_pb2
from gz.msgs10 import float_v_pb2

GRAVITY = 9.80665
# From iris_with_standoffs/model.sdf and the motor-model world.
MOTOR_CONSTANT = 8.54858e-06  # N / (rad/s)^2
MOMENT_CONSTANT = 0.016       # torque / thrust ratio (yaw drag)
MAX_OMEGA = 1100.0
# Control-channel order in BetaflightPlugin: rotor_3, rotor_0, rotor_1, rotor_2.
ROTOR_POSITIONS = [
    (-0.13, -0.20),  # rotor_3, rear right, cw
    (0.13, -0.22),   # rotor_0, front right, ccw
    (-0.13, 0.20),   # rotor_1, rear left, cw
    (0.13, 0.22),    # rotor_2, front left, ccw
]
ROTOR_SPIN = [-1.0, 1.0, -1.0, 1.0]  # cw = -1, ccw = +1


def _rotation(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(m: List[List[float]], v: List[float]) -> List[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def motor_wrench(cmds: List[float]) -> tuple[List[float], List[float]]:
    """Normalized motor commands -> body-frame (force, torque)."""
    thrusts = []
    for cmd in cmds[:4]:
        cmd = max(0.0, min(1.0, float(cmd)))
        omega = cmd * MAX_OMEGA
        thrusts.append(MOTOR_CONSTANT * omega * omega)
    total_thrust = sum(thrusts)
    force = [0.0, 0.0, total_thrust]
    tau_x = 0.0
    tau_y = 0.0
    tau_z = 0.0
    for (x, y), spin, thrust in zip(ROTOR_POSITIONS, ROTOR_SPIN, thrusts):
        tau_x += y * thrust
        tau_y += -x * thrust
        tau_z += MOMENT_CONSTANT * spin * thrust
    return force, [tau_x, tau_y, tau_z]


class RotorForceBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.node = transport.Node()
        topic = (
            f"/world/{args.world}/wrench/persistent"
            if args.persistent
            else f"/world/{args.world}/wrench"
        )
        self.pub = self.node.advertise(topic, entity_wrench_pb2.EntityWrench)
        if not self.pub.valid():
            raise SystemExit(f"failed to advertise {topic}")
        self.current_force = [0.0, 0.0, 0.0]
        self.current_torque = [0.0, 0.0, 0.0]
        self.entity_name = args.entity
        self.last_cmds = [0.0, 0.0, 0.0, 0.0]
        self.last_rpy = [0.0, 0.0, 0.0]
        self.have_telemetry = False

    def _publish(self, force: List[float], torque: List[float]) -> None:
        if self.args.persistent:
            # ApplyLinkWrench accumulates persistent wrenches.  Send the
            # difference between desired and already-applied force so the
            # total converges to the desired value without runaway.
            df = [force[i] - self.current_force[i] for i in range(3)]
            dt = [torque[i] - self.current_torque[i] for i in range(3)]
            if max(abs(v) for v in df + dt) < 1e-6:
                return
            self.current_force = list(force)
            self.current_torque = list(torque)
        else:
            df, dt = list(force), list(torque)

        msg = entity_wrench_pb2.EntityWrench()
        msg.entity.name = self.entity_name
        msg.entity.type = (
            entity_pb2.Entity.LINK
            if self.args.entity_type == "link"
            else entity_pb2.Entity.MODEL
        )
        msg.wrench.force.x = float(df[0])
        msg.wrench.force.y = float(df[1])
        msg.wrench.force.z = float(df[2])
        msg.wrench.torque.x = float(dt[0])
        msg.wrench.torque.y = float(dt[1])
        msg.wrench.torque.z = float(dt[2])
        self.pub.publish(msg)

    def telemetry_cb(self, msg: float_v_pb2.Float_V) -> None:
        data = list(msg.data)
        if len(data) < 8:
            return
        self.last_cmds = [max(0.0, min(1.0, data[i])) for i in range(4)]
        self.last_rpy = [data[5], data[6], data[7]]
        self.have_telemetry = True
        force_body, torque_body = motor_wrench(self.last_cmds)
        rot = _rotation(*self.last_rpy)
        force_world = _mat_vec(rot, force_body)
        torque_world = _mat_vec(rot, torque_body)
        self._publish(force_world, torque_world)

    def spin(self) -> None:
        if self.args.mode == "telemetry":
            self.node.subscribe(
                float_v_pb2.Float_V,
                self.args.telemetry_topic,
                self.telemetry_cb,
            )
            print(
                f"[rotor-force] telemetry mode: {self.args.telemetry_topic} "
                f"-> /world/{self.args.world}/wrench, entity={self.entity_name}",
                flush=True,
            )
            while True:
                time.sleep(0.5)
                if self.args.verbose and not self.have_telemetry:
                    print("[rotor-force] waiting for telemetry...", flush=True)
        else:
            total_mass = self.args.mass
            force = [0.0, 0.0, total_mass * GRAVITY]
            print(
                f"[rotor-force] hover mode: Fz={force[2]:.2f} N -> "
                f"/world/{self.args.world}/wrench, entity={self.entity_name}",
                flush=True,
            )
            if self.args.persistent:
                while True:
                    self._publish(force, [0.0, 0.0, 0.0])
                    time.sleep(0.05)
            while True:
                self._publish(force, [0.0, 0.0, 0.0])
                time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description="RotorForce bridge")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--entity", default="iris_with_Betaflight")
    parser.add_argument("--entity-type", choices=("model", "link"), default="model")
    parser.add_argument("--mode", choices=("telemetry", "hover"), default="telemetry")
    parser.add_argument("--telemetry-topic", default="/betaflight/telemetry")
    parser.add_argument("--persistent", action="store_true",
                        help="publish to /wrench/persistent (recommended for hover test)")
    parser.add_argument("--mass", type=float, default=1.6)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    RotorForceBridge(args).spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
