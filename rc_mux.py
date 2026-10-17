"""RC 融合层: 唯一的 MSP RC 写入者.

规则 (优先级从高到低):
  1. CRSF 静默 > crsf_timeout -> 完全停发 -> BF failsafe
  2. 追踪开关低位 -> 强制透传; 开关须先见过低位, 上升沿才有效 (C7)
  3. 接管检测 (仅替换模式): R/P/Y 偏中位>30% 或油门偏切入捕获值>150 -> 回透传+抑制
  4. 替换条件: 开关高且 TrackCmd 新鲜 (<cmd_timeout); 指令过期自动回透传
  5. 进入替换 0.5s 斜坡; 退出立即
  6. 无 CRSF -> BENCH 模式 (键盘切换, base_throttle)
AUX1-4 (含 ARM) 永远透传飞手.
FC 模式 (channel_source): AUX 只回放飞手缓存值; RC 缓存过期 (>rc_stale_timeout)
或不足 8 通道时不发任何 MSP RC 帧, 交给飞控原生 failsafe, 绝不猜 AUX.
BENCH 模式是唯一会主动置 AUX1=2000 的路径 (仅拆桨台架).
"""
import time
import threading
from dataclasses import dataclass


@dataclass
class TrackCmd:
    pitch: int
    yaw: int
    throttle: int
    roll: int = 1500
    t: float = 0.0


def _clamp(v, lo=1000, hi=2000):
    return max(lo, min(hi, int(v)))


class RCMux:
    RPY_DEADBAND = 0.30
    THR_TAKEOVER_DELTA = 150
    RAMP_TIME = 0.5

    def __init__(self, msp, crsf=None, switch_channel=None,
                 base_throttle=1100, crsf_timeout=0.5, cmd_timeout=0.5,
                 rate_hz=50, channel_source=None, rc_stale_timeout=0.5):
        self.msp = msp
        self.crsf = crsf
        self.switch_channel = switch_channel
        self.base_throttle = base_throttle
        self.crsf_timeout = crsf_timeout
        self.cmd_timeout = cmd_timeout
        self.rate_hz = rate_hz
        self._channel_source = channel_source  # ()->(channels, age) for FC mode
        # FC mode: when the MSP_RC cache is older than this, send NO MSP RC
        # frame at all and let the FC's native failsafe take over.  Never
        # replay stale pilot values or guess AUX.
        self.rc_stale_timeout = max(0.0, float(rc_stale_timeout))
        self.on_engage = None
        self.on_disengage = None

        self.mode = "PASSTHROUGH"
        self.engaged_throttle = base_throttle
        self._cmd = None
        self._auto = False
        self._switch_prev = False
        self._switch_seen_low = False
        self._inhibit = False
        self._ramp_start = 0.0
        self._was_substituting = False
        self._running = False
        self._thread = None

    def publish_cmd(self, cmd):
        self._cmd = cmd

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, park=False):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if park:
            try:
                self.msp.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
            except Exception:
                pass

    def _tick(self, now):
        if self.crsf is not None:
            self._tick_crsf(now)
            return
        if self._channel_source is not None:
            self._tick_fc_mode(now)
            return
        self._tick_bench(now)

    # ── CRSF 本地直连模式 ──────────────────────────────────────────────

    def _tick_crsf(self, now):
        ch = self.crsf.channels()
        if ch is None or self.crsf.age(now) > self.crsf_timeout:
            if self.mode != "SILENT":
                self.mode = "SILENT"
                self._switch_seen_low = False
                if self._auto:
                    self._exit_auto()
            return
        if self.mode == "SILENT":
            self.mode = "PASSTHROUGH"

        self._update_switch(ch)
        if self._auto and self._takeover(ch):
            self._exit_auto()
            self._inhibit = True

        rpyt = self._select_rpyt(now, ch)
        aux = ch[4:8]
        self.msp.set_rc(rpyt[0], rpyt[1], rpyt[2], rpyt[3],
                        aux[0], aux[1], aux[2], aux[3])

    # ── FC 模式: 接收机在飞控上, X5 通过 MSP_GET_RC 读通道 ─────────────

    def _tick_fc_mode(self, now):
        """FC 架构: 开关/接管检测走 MSP RC 缓存, AUX 永远只透传飞手.

        RPYT 在 AUTO 时用追踪值, 否则用飞手值. AUX 从不写固定值 (旧代码在
        这里硬编码 AUX3=2000, 会让伴飞逻辑自己去拨 BOXMSPOVERRIDE).
        RC 缓存过期时不发任何 MSP RC 帧: 让飞控原生 failsafe 接管.
        """
        rc_data, rc_age = self._channel_source()
        if rc_data is None or rc_age > self.rc_stale_timeout or len(rc_data) < 8:
            self._fc_stale()
            return

        ch = list(rc_data[:16])
        self.mode = "PASSTHROUGH"  # FC 架构无 SILENT, failsafe 由飞控原生 CRSF 保证

        self._update_switch(ch)
        if self._auto and self._takeover(ch):
            self._exit_auto()
            self._inhibit = True

        if self._auto:
            rpyt = self._select_rpyt(now, ch)
        else:
            # 透传飞手 RPYT — AUX3 切高位时 FC 立刻用 MSP 值, 必须无缝衔接
            rpyt = (ch[0], ch[1], ch[2], ch[3])

        # AUX1..4 (16 通道时还包括 AUX5..12) 只透传飞手缓存值, 绝不改写成
        # ARM / ANGLE / BOXMSPOVERRIDE / engage 的指令.
        if hasattr(self.msp, "send_rc_16"):
            channels = list(rpyt) + ch[4:16]
            if len(channels) < 16:
                channels.extend([1500] * (16 - len(channels)))
            self.msp.send_rc_16(channels)
        else:
            aux = ch[4:8]
            self.msp.set_rc(rpyt[0], rpyt[1], rpyt[2], rpyt[3],
                            aux[0], aux[1], aux[2], aux[3])

    def _fc_stale(self):
        """RC 缓存过期: 不发 MSP RC 帧, 让 FC failsafe; 退出 AUTO 并上锁."""
        if self.mode != "FC_STALE":
            self.mode = "FC_STALE"
            self._was_substituting = False
            self._switch_prev = False
            self._switch_seen_low = False
            if self._auto:
                self._exit_auto()

    def _update_switch(self, ch):
        if self.switch_channel is None or len(ch) <= self.switch_channel:
            return
        raw = ch[self.switch_channel]
        if raw < 1300:
            sw = False
            self._switch_seen_low = True
            self._inhibit = False
        elif raw > 1700:
            sw = True
        else:
            sw = self._switch_prev
        rising = sw and not self._switch_prev
        self._switch_prev = sw
        if not sw and self._auto:
            self._exit_auto()
        elif (rising and not self._auto and self._switch_seen_low
              and not self._inhibit):
            self._enter_auto(ch)

    def _takeover(self, ch):
        for idx in (0, 1, 3):
            if abs(ch[idx] - 1500) > 500 * self.RPY_DEADBAND:
                return True
        return abs(ch[2] - self.engaged_throttle) > self.THR_TAKEOVER_DELTA

    def _select_rpyt(self, now, ch):
        cmd = self._cmd
        fresh = (self._auto and cmd is not None
                 and (now - cmd.t) <= self.cmd_timeout)
        if not fresh:
            self._was_substituting = False
            self.mode = "PASSTHROUGH"
            return (ch[0], ch[1], ch[2], ch[3])
        if not self._was_substituting:
            self._ramp_start = now
            self._was_substituting = True
        self.mode = "AUTO"
        blend = min(1.0, (now - self._ramp_start) / self.RAMP_TIME)
        roll = 1500 + (cmd.roll - 1500) * blend
        pitch = 1500 + (cmd.pitch - 1500) * blend
        yaw = 1500 + (cmd.yaw - 1500) * blend
        thr = self.engaged_throttle + (cmd.throttle - self.engaged_throttle) * blend
        return (_clamp(roll), _clamp(pitch), _clamp(thr), _clamp(yaw))

    def _enter_auto(self, ch):
        self._auto = True
        self.engaged_throttle = ch[2]
        self._was_substituting = False
        if self.on_engage:
            self.on_engage(self.engaged_throttle)

    def _exit_auto(self):
        self._auto = False
        self._was_substituting = False
        if self.on_disengage:
            self.on_disengage()

    def bench_toggle(self):
        if self.crsf is not None:
            return self._auto
        if self._auto:
            self._exit_auto()
        else:
            self._auto = True
            self.engaged_throttle = self.base_throttle
            self._was_substituting = False
            if self.on_engage:
                self.on_engage(self.base_throttle)
        return self._auto

    def _tick_bench(self, now):
        self.mode = "BENCH"
        cmd = self._cmd
        fresh = (self._auto and cmd is not None
                 and (now - cmd.t) <= self.cmd_timeout)
        if fresh:
            if not self._was_substituting:
                self._ramp_start = now
                self._was_substituting = True
            blend = min(1.0, (now - self._ramp_start) / self.RAMP_TIME)
            pitch = 1500 + (cmd.pitch - 1500) * blend
            yaw = 1500 + (cmd.yaw - 1500) * blend
            thr = self.engaged_throttle + (cmd.throttle - self.engaged_throttle) * blend
            self.msp.set_rc(_clamp(cmd.roll), _clamp(pitch), _clamp(thr),
                            _clamp(yaw), 2000, 1500, 1500, 1500)
        elif self._auto:
            self._was_substituting = False
            self.msp.set_rc(1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)
        else:
            self._was_substituting = False
            self.msp.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)

    def _run(self):
        period = 1.0 / self.rate_hz
        next_t = time.monotonic()
        last_err = 0.0
        while self._running:
            now = time.monotonic()
            try:
                self._tick(now)
            except Exception as e:
                if now - last_err > 1.0:
                    print(f"[RCMux] tick error: {e!r}")
                    last_err = now
            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()
