# Firmware baseline for Pi5 sim2real

This directory is metadata only. The firmware source stays in the user's
Betaflight checkout; do not copy the full tree into BF_Anti.

## Frozen single baseline

- Base: Betaflight branch release260904, commit 44482a4c7f.
- Working branch: sim2real/2026-10-17, commit 5ce40b7691, tag
  sim2real-2026-10-17.
- Isolated worktree: /tmp/bf_sim2real.
- release260904 was not modified by this work.

The SITL/offboard WIP is committed as-is. We are not producing a separate
"flight" branch and we are not claiming master-equivalent control laws.
The flight gate is narrower:

1. **Used-path regression** - ARM + ANGLE + BOXMSPOVERRIDE + companion RPYT
   must not be changed by the WIP. Static audit is in
   docs/sim2real/firmware-used-path-audit.md. The dynamic gate is a same-FC,
   props-off A/B: clean release260904 build vs this baseline, same RC sequence,
   compare MSP_MOTOR point by point (tolerance 1%).
2. **Protocol contract** - MSP frame/CRC, 16-channel AETR, yaw units,
   altitude/vario, status/armed, analog length, board info, motor readback.
   The contract and conformance tool live in BF_Anti.

## Build

SITL (already built from this baseline):

    cd ~/betaflight
    make CONFIG=SITL_GAZEBO -j8
    # ELF: obj/main/betaflight_SITL_SITL_GAZEBO.elf

Real FC target is unknown until MSP_BOARD_INFO is read from the board.
Do not flash a guessed target. When the board is connected:

    python3 scripts/fc_identify.py --port /dev/ttyACM0
    # match the 4-character board identifier to a target in src/platform/STM32/target

## Files changed by the WIP on the used path

Only two used-path files differ from release260904:

- src/main/flight/mixer.c - thrust-domain compensation and thrust-priority
  desaturation, both gated to ALT_HOLD / POS_HOLD / GPS_RESCUE. ANGLE +
  BOXMSPOVERRIDE does not enter those gates.
- src/main/fc/core.c - ALT_HOLD/POS_HOLD entry may skip the pilot throttle
  raise when positionOffboardIsActive(). It is still inside the
  ALT_HOLD/POS_HOLD mode entry and does not run in the first-flight profile.

pid.c, imu.c, controlrate.c, motors.c and rc.c are byte identical to
release260904.
