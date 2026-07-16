#!/usr/bin/env python3
"""
PAL 视频帧生成 → SPI 发给 Pico 协处理器
RDK X5 SPI Master → Pico SPI Slave → PIO+DMA → GPIO → 视频
"""

import spidev, numpy as np, threading, time, sys

# ─── PAL 8 MHz 编码 (每 byte = 1µs) ────────────────────────────────────
SPI_HZ        = 16_000_000    # SPI 传给 Pico 的速度
LINE_BYTES    = 64             # 64 bytes/line = 64 µs
SYNC_BYTES    = 5
BACK_PORCH    = 6
FRONT_PORCH   = 2
ACTIVE_BYTES  = 51             # 408 pixels
ACTIVE_PIXELS = 408
ACTIVE_LINES  = 576
LINES_FRAME   = 625
VSYNC_LINES   = 6
VBLANK_TOP    = 16
VBLANK_BOTTOM = 27
FRAME_BYTES   = LINE_BYTES * LINES_FRAME  # 40000
FPS           = 25

assert SYNC_BYTES + BACK_PORCH + ACTIVE_BYTES + FRONT_PORCH == LINE_BYTES

# ─── Bayer 抖动 ────────────────────────────────────────────────────────
_BAYER = np.array([
    [  4,192, 51,239, 16,204, 63,251],[129, 67,177,114,141, 78,188,126],
    [ 35,224, 20,208, 47,235, 31,220],[161, 98,145, 82,173,110,157, 94],
    [ 12,200, 59,247,  8,196, 55,243],[137, 75,184,122,133, 71,181,118],
    [ 43,231, 27,216, 39,228, 24,212],[169,106,153, 90,165,102,149, 86],
], dtype=np.uint8)
_THR = {}
def _t(h,w):
    k=(h,w)
    if k not in _THR: _THR[k]=np.tile(_BAYER,(h//8+1,w//8+1))[:h,:w]
    return _THR[k]
def dither(g):
    return np.where(g>_t(*g.shape),255,0).astype(np.uint8)

# ─── PAL 帧缓冲区 ──────────────────────────────────────────────────────
class PALFrame:
    A_START = SYNC_BYTES + BACK_PORCH       # 11
    A_END   = SYNC_BYTES + BACK_PORCH + ACTIVE_BYTES  # 62
    F_ACTIVE = VSYNC_LINES + VBLANK_TOP     # 22
    F_END    = F_ACTIVE + ACTIVE_LINES      # 598

    def __init__(self):
        self._buf = bytearray(FRAME_BYTES)
        self._init_fixed()

    def _init_fixed(self):
        # VSYNC: 6行宽脉冲 (全零)
        # blank: 行同步+后肩全零, 有效区用 0x01 防止假同步
        blank = bytearray(LINE_BYTES)
        for i in range(self.A_START, self.A_END): blank[i] = 0x01
        bb = bytes(blank)
        for i in range(VSYNC_LINES, self.F_ACTIVE): self._set(i, bb)
        for i in range(self.F_END, LINES_FRAME): self._set(i, bb)

    def _set(self, idx, data):
        self._buf[idx*LINE_BYTES:(idx+1)*LINE_BYTES] = data

    def update_pixels(self, bitmap):
        if bitmap.shape != (ACTIVE_LINES, ACTIVE_PIXELS):
            raise ValueError
        # packbits LITTLE: pixel[0] → bit0 → PIO LSB 先出
        packed = np.packbits(bitmap.ravel(), bitorder='little')
        packed[packed == 0] = 0x01
        buf2d = np.frombuffer(self._buf, dtype=np.uint8).reshape(LINES_FRAME, LINE_BYTES)
        buf2d[self.F_ACTIVE:self.F_END, self.A_START:self.A_END] = packed.reshape(ACTIVE_LINES, ACTIVE_BYTES)

    def buffer(self):
        return bytes(self._buf)

# ─── 视频输出 (SPI → Pico) ─────────────────────────────────────────────
class PALVideo:
    def __init__(self):
        self._pal = PALFrame()
        self._spi = spidev.SpiDev()
        self._spi.open(1, 0)
        self._spi.max_speed_hz = SPI_HZ
        self._spi.mode = 0
        self._spi.bits_per_word = 8
        self._spi.lsbfirst = False
        self._current = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._n = 0
        self._drop = 0
        self._update_seq = 0   # update_frame_bgr 调用次数
        self._last_sent_seq = 0

    def update_frame_bgr(self, bgr):
        try:
            import cv2
            if bgr.shape[1]!=ACTIVE_PIXELS or bgr.shape[0]!=ACTIVE_LINES:
                bgr = cv2.resize(bgr, (ACTIVE_PIXELS, ACTIVE_LINES))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except ImportError:
            bgr = bgr[::max(1,bgr.shape[0]//ACTIVE_LINES),
                      ::max(1,bgr.shape[1]//ACTIVE_PIXELS)][:ACTIVE_LINES,:ACTIVE_PIXELS]
            gray = (bgr[...,2].astype(np.float32)*0.299+
                    bgr[...,1].astype(np.float32)*0.587+
                    bgr[...,0].astype(np.float32)*0.114).astype(np.uint8)
        binary = dither(gray)
        self._pal.update_pixels(binary)
        with self._lock:
            self._current = self._pal.buffer()
            self._update_seq += 1

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2)
        self._spi.close()

    @property
    def stats(self): return (self._n, self._drop)

    def _run(self):
        blank = bytes(FRAME_BYTES)
        for _ in range(5):
            self._spi.writebytes2(blank)
            time.sleep(0.04)
        t_next = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now < t_next:
                time.sleep(max(0, t_next-now-0.005))
                while time.monotonic() < t_next: pass
            else: t_next = now + 0.04
            with self._lock:
                cur = self._current
                cur_seq = self._update_seq
            if cur is None: cur = blank
            self._spi.writebytes2(cur)
            self._n += 1
            if cur_seq == self._last_sent_seq:
                self._drop += 1   # 没有新帧 → 重复发送旧帧
            self._last_sent_seq = cur_seq
            t_next += 0.04
        self._spi.writebytes2(blank)

# ─── 测试 ───────────────────────────────────────────────────────────────
def test_pattern():
    import cv2
    print(f"PAL 8MHz {ACTIVE_PIXELS}x{ACTIVE_LINES} → SPI → Pico")
    pattern = np.zeros((ACTIVE_LINES, ACTIVE_PIXELS), dtype=np.uint8)
    cs = 16
    for y in range(0, ACTIVE_LINES, cs):
        for x in range(0, ACTIVE_PIXELS, cs):
            if ((x//cs)+(y//cs))%2==0: pattern[y:y+cs,x:x+cs]=255
    pal = PALVideo()
    pal.start()
    bar = 0
    print("调电位器, Ctrl+C 停")
    try:
        while True:
            f = pattern.copy()
            bar = (bar+2)%ACTIVE_PIXELS
            f[:,bar:bar+4]=255
            pal.update_frame_bgr(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
            time.sleep(0.04)
    except KeyboardInterrupt: pass
    pal.stop()
    print(f"\n{pal.stats[0]} 帧")

if __name__ == "__main__":
    test_pattern()
