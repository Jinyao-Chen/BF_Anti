#!/bin/bash
# BF_Anti 反无人机拦截 SITL 启动器
#
# 复用 ~/betaflight (release260904) 的 SITL_GAZEBO 固件与 Gazebo Harmonic
# BetaflightPlugin，只把世界文件换成 BF_Anti 的固定靶机场景。
#
# 用法:
#   ./sim/launch_sitl.sh              # GUI
#   ./sim/launch_sitl.sh --headless   # 无界面，适合自动化测试
#
# 环境变量:
#   BF_ROOT=/path/to/betaflight          # 默认 $HOME/betaflight
#   BF_ANTI_WORLD=/path/to/world.sdf     # 默认真实飞机靶机 world
#   BF_ANTI_CONFIG=/path/to/config.txt   # 覆盖 SITL EEPROM 配置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BF_ROOT="${BF_ROOT:-$HOME/betaflight}"
GAZEBO_DIR="$BF_ROOT/src/platform/SIMULATOR/gazebo"
PLUGIN_SRC="$GAZEBO_DIR/betaflight_gazebo"
PLUGIN_BUILD="$GAZEBO_DIR/build"
PLUGIN_LIB="$PLUGIN_BUILD/lib"
CONFIG_DIR="$GAZEBO_DIR/config"
BF_BIN="$BF_ROOT/obj/main/betaflight_SITL_SITL_GAZEBO.elf"
WORLD_FILE="${BF_ANTI_WORLD:-$SCRIPT_DIR/worlds/bf_anti_intercept_real.sdf}"
# PX4 Gazebo Harmonic models are used by the real-aircraft target.  Optional:
# machines without PX4-Autopilot simply won't have this path and the default
# real-target world will fail to load, so users can point BF_ANTI_WORLD at the
# legacy box world instead.
PX4_GZ_MODELS="${PX4_GZ_MODELS:-$HOME/PX4-Autopilot/Tools/simulation/gz/models}"

# Motor-model / force-only worlds use different hover baselines.
# The normal BF_Anti world defaults to the validated release offboard config.
# Direct-RC tracking experiments can opt in explicitly with:
#   BF_ANTI_CONFIG=$SCRIPT_DIR/config/sitl_gazebo_tracking.txt ./sim/launch_sitl.sh
if [[ "$WORLD_FILE" == *light* ]]; then
    CONFIG_FILE="${BF_ANTI_CONFIG:-$SCRIPT_DIR/config/sitl_gazebo_p_strike_light.txt}"
elif [[ "$WORLD_FILE" == *p_strike* ]]; then
    CONFIG_FILE="${BF_ANTI_CONFIG:-$SCRIPT_DIR/config/sitl_gazebo_p_strike.txt}"
elif [[ "$WORLD_FILE" == *motor* ]]; then
    CONFIG_FILE="${BF_ANTI_CONFIG:-$SCRIPT_DIR/config/sitl_gazebo_motor.txt}"
elif [[ "$WORLD_FILE" == *force* ]]; then
    CONFIG_FILE="${BF_ANTI_CONFIG:-$SCRIPT_DIR/config/sitl_gazebo_force.txt}"
else
    CONFIG_FILE="${BF_ANTI_CONFIG:-$CONFIG_DIR/sitl_gazebo_config.txt}"
fi

HEADLESS=0
for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -d "$BF_ROOT" ]; then
    echo "[bf-anti-sim] ERROR: BF_ROOT not found: $BF_ROOT" >&2
    echo "[bf-anti-sim] Set BF_ROOT=/path/to/betaflight and retry." >&2
    exit 1
fi

echo "[bf-anti-sim] building Betaflight SITL_GAZEBO (incremental)..."
make -C "$BF_ROOT" CONFIG=SITL_GAZEBO -j"$(nproc)"

if [ ! -f "$PLUGIN_LIB/libBetaflightPlugin.so" ]; then
    echo "[bf-anti-sim] building BetaflightPlugin..."
    if [ -f /opt/ros/humble/setup.bash ]; then
        set +u
        # shellcheck disable=SC1091
        source /opt/ros/humble/setup.bash
        set -u
    fi
    cmake -S "$PLUGIN_SRC" -B "$PLUGIN_BUILD" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PLUGIN_BUILD"
    cmake --build "$PLUGIN_BUILD" -j"$(nproc)"
    cmake --install "$PLUGIN_BUILD"
fi

# Always regenerate EEPROM from the selected config.  The LiftDrag and
# motor-model worlds need different hover/PID values, and switching worlds must
# not silently reuse the previous world's EEPROM.
echo "[bf-anti-sim] writing SITL EEPROM from $CONFIG_FILE..."
(cd "$CONFIG_DIR" && "$BF_BIN" --config "$CONFIG_FILE")

if ! command -v gz >/dev/null 2>&1; then
    echo "[bf-anti-sim] ERROR: Gazebo 'gz' not found in PATH." >&2
    exit 1
fi

export GZ_SIM_SYSTEM_PLUGIN_PATH="$PLUGIN_LIB:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
if [ -d "$PX4_GZ_MODELS" ]; then
    export GZ_SIM_RESOURCE_PATH="$GAZEBO_DIR/models:$SCRIPT_DIR/models:$PX4_GZ_MODELS:${GZ_SIM_RESOURCE_PATH:-}"
else
    export GZ_SIM_RESOURCE_PATH="$GAZEBO_DIR/models:$SCRIPT_DIR/models:${GZ_SIM_RESOURCE_PATH:-}"
fi

GZ_PID=""
cleanup() {
    echo "[bf-anti-sim] stopping..."
    kill "${BF_PID:-}" 2>/dev/null || true
    kill "${GZ_PID:-}" 2>/dev/null || true
    wait 2>/dev/null || true
    pkill -f "betaflight_SITL_SITL_GAZEBO.elf" 2>/dev/null || true
    pkill -f "gz sim.*bf_anti_intercept" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Start SITL first: the Gazebo plugin's socket_out.Connect() to the SITL FDM
# port is not retried.  If Gazebo wins the startup race the firmware can sit
# without an FDM stream, the RC failsafe latches, and the next arm attempt is
# rejected.  SITL-first removes that race.
echo "[bf-anti-sim] starting Betaflight SITL..."
(cd "$CONFIG_DIR" && "$BF_BIN" --ip 127.0.0.1) &
BF_PID=$!

PORT_UP=0
for _ in $(seq 1 40); do
    if (exec 3<>/dev/tcp/127.0.0.1/5761) 2>/dev/null; then
        PORT_UP=1
        break
    fi
    if ! kill -0 "$BF_PID" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if [ "$PORT_UP" = "1" ]; then
    echo "[bf-anti-sim] SITL is up (MSP 127.0.0.1:5761)."
else
    echo "[bf-anti-sim] WARNING: MSP port did not open; see firmware output above." >&2
fi

if ! command -v gz >/dev/null 2>&1; then
    echo "[bf-anti-sim] ERROR: Gazebo 'gz' not found in PATH." >&2
    exit 1
fi

if [ "$HEADLESS" = "1" ]; then
    echo "[bf-anti-sim] starting Gazebo (headless)..."
    gz sim -r -s -v 3 "$WORLD_FILE" &
else
    echo "[bf-anti-sim] starting Gazebo (GUI)..."
    gz sim -r -v 3 "$WORLD_FILE" &
fi
GZ_PID=$!

sleep 3
if ! kill -0 "$GZ_PID" 2>/dev/null; then
    echo "[bf-anti-sim] ERROR: Gazebo exited immediately." >&2
    exit 1
fi

# Give the plugin a moment to establish the FDM stream before the user starts
# sending RC.  The firmware failsafe/arming state machine is cleanest when the
# RC stream only appears after FDM is flowing.
sleep 2
echo "[bf-anti-sim] target_drone (real x500) at (30, 0, 10) in Gazebo world coordinates."

while kill -0 "$BF_PID" 2>/dev/null && kill -0 "$GZ_PID" 2>/dev/null; do
    sleep 2
done
