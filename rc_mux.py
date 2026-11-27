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
                 rate_hz=50, channel_source=None, rc_stale_timeout=0.5,
                 mode_channel=None, arm_channel=None,
                 stick_takeover=True, auto_engage=True,
                 override_channel=None, override_range=None):
        self.msp = msp
        self.crsf = crsf
        self.switch_channel = switch_channel
        # Three-position mode switch (low=force manual, mid=auto, high=strike).
        # When set it supersedes the legacy two-position switch_channel.
        self.mode_channel = mode_channel
        self.override_channel = override_channel
        self.override_range = override_range
        self.arm_channel = arm_channel
        self.stick_takeover = bool(stick_takeover)
        self.auto_engage = bool(auto_engage)
        self._mode_pos = 0          # 0 manual, 1 auto, 2 strike
        self._mode_prev = 0
        self._safe_hover_thr = int(base_throttle)
        self._mode_raw = 1500
        self._mode_seen_low = False
        self._reset_needed = False
        self._auto_request = False
        self._arm_prev = None
        # Optional (start, end) PWM range discovered from MSP_MODE_RANGES.
        # None keeps the legacy >1700 threshold (bench/SITL).
        self.arm_range = None
        # Called as fn(safe_throttle) -> (roll, pitch, throttle, yaw) when the
        # override switch is still high outside AUTO.  In ACRO, 1500 holds the
        # current dive attitude, so the app supplies a level-attitude command.
        self.level_rpyt_fn = None
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

    def _arm_active(self, ch) -> bool:
        if self.arm_channel is None or len(ch) <= self.arm_channel:
            return True
        raw = int(ch[self.arm_channel])
        rng = getattr(self, "arm_range", None)
        if rng is not None:
            try:
                return int(rng[0]) <= raw <= int(rng[1])
            except (TypeError, ValueError, IndexError):
                pass
        return raw > 1700

    def _override_active(self, ch) -> bool:
        """True when the FC's BOXMSPOVERRIDE range is currently active.

        In the 3-position mode configuration the mode channel carries both the
        pilot intent and the FC override range.  In the legacy separate-switch
        configuration the override channel/range must be used instead; using
        only mode_channel here silently samples MSP_RC echo values and
        disables the safe level-attitude path.
        """
        if self.override_channel is not None:
            idx = int(self.override_channel)
            if not (0 <= idx < len(ch)):
                # An explicitly configured override channel that is missing
                # must fail closed, not silently fall back to the mode switch.
                return False
            raw = int(ch[idx])
            if self.override_range is not None:
                try:
                    return bool(int(self.override_range[0]) <= raw
                                <= int(self.override_range[1]))
                except (TypeError, ValueError, IndexError):
                    return False
            return raw > 1700
        return bool(self.mode_channel is not None and self._mode_pos > 0)

    def _safe_override_rpyt(self, ch):
        """Command when BOXMSPOVERRIDE is still on but AUTO is not active."""
        if not self._arm_active(ch):
            return (1500, 1500, 1000, 1500)
        cb = getattr(self, "level_rpyt_fn", None)
        if callable(cb):
            try:
                values = tuple(int(v) for v in cb(int(self._safe_hover_thr)))
                if len(values) >= 4:
                    return values[:4]
            except Exception:
                pass
        return (1500, 1500, int(self._safe_hover_thr), 1500)

    def request_auto(self, want):
        """Ask the mux to engage.  The mux still owns the final decision.

        The app calls this every control tick with the safety supervisor's
        verdict.  A target loss simply drops the request; re-acquisition can
        engage again without cycling the switch.  Force-manual and disarmed
        states are enforced in _update_mode and can never be overridden here.
        """
        self._auto_request = bool(want)

    def mode_position(self):
        return int(self._mode_pos)

    def mode_name(self):
        return ("MANUAL", "AUTO", "STRIKE")[max(0, min(2, self._mode_pos))]

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
                # AUX must be low, not 1500: a mode range that happens to span
                # mid-stick (e.g. ANGLE/PREARM) must never activate on park.
                self.msp.set_rc(1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000)
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

        if self.mode_channel is not None:
            self._update_mode(ch)
        else:
            self._update_switch(ch)
        # Only sample pilot throttle when BOXMSPOVERRIDE is OFF so MSP_RC echo
        # values never overwrite the safe hover reference.
        override_active = self._override_active(ch)
        arm_active = self._arm_active(ch)
        if not arm_active:
            self._safe_hover_thr = min(1700, int(self.base_throttle))
        elif not override_active:
            # Never let a low/dropped pilot stick or a full-climb stick become
            # the fail-safe throttle.
            max_safe = min(1700, int(self.base_throttle) + 150)
            self._safe_hover_thr = _clamp(
                min(max(int(ch[2]), int(self.base_throttle)), max_safe)
            )
        if self._auto and self.stick_takeover and self._takeover(ch):
            self._exit_auto()
            self._inhibit = True

        if self._auto:
            rpyt = self._select_rpyt(now, ch, override_active=override_active)
        elif override_active:
            # Betaflight 4.5 MSP_RC returns overridden RPYT while BOXMSPOVERRIDE
            # is active; echoing ch[0..3] back latches the last dive/throttle.
            # In ACRO, (1500,1500) is zero body rate, not level flight: the app
            # callback computes a level-attitude command from live rpy.
            rpyt = self._safe_override_rpyt(ch)
        else:
            # Physical manual mode: passthrough pilot RPYT seamlessly.
            rpyt = (ch[0], ch[1], ch[2], ch[3])

        # AUX1..4 (16 通道时还包括 AUX5..12) 只透传飞手缓存值, 绝不改写成
        # ARM / ANGLE / BOXMSPOVERRIDE / engage 的指令.
        if hasattr(self.msp, "send_rc_16"):
            channels = list(rpyt) + ch[4:16]
            if len(channels) < 16:
                # Missing AUX must mean "low/disarmed", never a 1500 mid value
                # that could activate an unconfigured mode range.
                channels.extend([1000] * (16 - len(channels)))
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
            self._mode_seen_low = False
            self._mode_pos = 0
            self._safe_hover_thr = min(1700, int(self.base_throttle))
            self._inhibit = True
            # On recovery, a switch already LOW has no falling edge; remember
            # to clear a latched safety fault when the pilot next visits LOW.
            self._reset_needed = True
            if self._auto:
                self._exit_auto()

    def _update_mode(self, ch):
        """Three-position pilot switch + manual arm lockout.

        low  = FORCE MANUAL: never auto, even if the app requests it.
        mid  = AUTO: engage when the app says the target/safety gate is green.
        high = STRIKE: same engagement, weapons-hot intent.

        The companion never writes ARM.  A low ARM channel forces the mux out
        of AUTO and inhibits re-engagement; the next manual arm edge is the
        pilot's explicit consent and clears the inhibit.  This is why the
        aircraft never arms itself but can still auto-strike right after the
        pilot arms with a target already in view.
        """
        if len(ch) <= self.mode_channel:
            return
        raw = int(ch[self.mode_channel])
        self._mode_raw = raw
        prev_pos = self._mode_pos
        reset_called = False
        if raw < 1300:
            pos = 0
            self._mode_seen_low = True
            self._inhibit = False
            if self._reset_needed:
                self._reset_needed = False
                if self.on_disengage:
                    self.on_disengage()
                    reset_called = True
        elif raw > 1700:
            pos = 2
        else:
            pos = 1
        self._mode_pos = pos

        if self.arm_channel is not None and len(ch) > self.arm_channel:
            arm_high = self._arm_active(ch)
            if not arm_high:
                # Disarmed: kill auto immediately and do not let a stale
                # request re-engage while the mode switch is still high.
                if self._auto:
                    self._exit_auto()
                elif self._arm_prev is not False and self.on_disengage:
                    # Manual disarm is also the pilot's reset edge for a
                    # latched safety fault.
                    self.on_disengage()
                    reset_called = True
                self._inhibit = True
            elif self._arm_prev is False:
                # Manual arm edge: pilot consent.  Clear the disarm inhibit but
                # preserve C7: the mode switch must still have been seen LOW
                # at least once, even if it was already high at arming.
                self._inhibit = False
            self._arm_prev = bool(arm_high)

        if pos == 0:
            if self._auto:
                self._exit_auto()
            elif prev_pos != 0 and self.on_disengage and not reset_called:
                # Pilot deliberately selected FORCE MANUAL.  This is also
                # the reset edge for a latched safety fault, so AUTO can be
                # re-entered after the switch is cycled low -> mid/high.
                self.on_disengage()
            self._mode_prev = pos
            return
        self._mode_prev = pos
        if self._auto:
            return
        if (self.auto_engage and self._auto_request and self._mode_seen_low
                and not self._inhibit):
            self._enter_auto(ch)

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
        prev_sw = self._switch_prev
        rising = sw and not prev_sw
        self._switch_prev = sw
        if not sw and self._auto:
            self._exit_auto()
        elif not sw and prev_sw and self.on_disengage:
            # Allow pilot to clear a latched safety fault by cycling switch low.
            self.on_disengage()
        elif (rising and not self._auto and self._switch_seen_low
              and not self._inhibit):
            self._enter_auto(ch)

    def _takeover(self, ch):
        for idx in (0, 1, 3):
            if abs(ch[idx] - 1500) > 500 * self.RPY_DEADBAND:
                return True
        return abs(ch[2] - self.engaged_throttle) > self.THR_TAKEOVER_DELTA

    def _select_rpyt(self, now, ch, override_active: bool = False):
        cmd = self._cmd
        fresh = (self._auto and cmd is not None
                 and (now - cmd.t) <= self.cmd_timeout)
        if not fresh:
            self._was_substituting = False
            self.mode = "PASSTHROUGH"
            if override_active:
                return self._safe_override_rpyt(ch)
            return (ch[0], ch[1], ch[2], ch[3])
        if not self._was_substituting:
            self._ramp_start = now
            # In FC override mode, ch[2] is the MSP echo, so anchor to the
            # last throttle sampled while BOXMSPOVERRIDE was OFF.
            self.engaged_throttle = (
                self._safe_hover_thr if override_active else ch[2]
            )
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
        override_active = self._override_active(ch)
        # While BOXMSPOVERRIDE is active, MSP_RC echoes our own last frame;
        # anchor to the throttle sampled before override instead.
        self.engaged_throttle = (
            self._safe_hover_thr if override_active else int(ch[2])
        )
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
