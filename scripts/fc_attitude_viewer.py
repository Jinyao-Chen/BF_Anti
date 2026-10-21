#!/usr/bin/env python3
"""Live MSP attitude viewer for FC sign verification.

Runs on the Pi (the FC is on /dev/ttyACM0) and serves a small browser page:

    python3 scripts/fc_attitude_viewer.py --port /dev/ttyACM0 --http-port 8090
    # browse http://<pi-ip>:8090/

The page shows raw roll/pitch/yaw in degrees and a simple artificial horizon.
It also writes a CSV sample log under build/fc/ for evidence.  No RC writes,
no arming, no motor output.

This is intentionally read-only.  Use it to tilt the FC by hand and verify
axis signs before any flight or motor test.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from msp import MSP_CMD, decode_attitude  # noqa: E402
from onboard.config import FcConfig  # noqa: E402
from onboard.fc.link import make_link  # noqa: E402


class AttitudeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.t = 0.0
        self.seq = 0
        self.samples = 0
        self.errors = 0
        self.last_error = ""

    def update(self, rpy: tuple[float, float, float]) -> None:
        with self._lock:
            self.roll, self.pitch, self.yaw = map(float, rpy)
            self.t = time.monotonic()
            self.seq += 1
            self.samples += 1
            self.last_error = ""

    def error(self, text: str) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = str(text)[:300]

    def snapshot(self) -> dict:
        with self._lock:
            age = (time.monotonic() - self.t) if self.t else None
            return {
                "roll_deg": round(self.roll, 2),
                "pitch_deg": round(self.pitch, 2),
                "yaw_deg": round(self.yaw, 2),
                "age_ms": None if age is None else round(age * 1000.0, 1),
                "seq": self.seq,
                "samples": self.samples,
                "errors": self.errors,
                "last_error": self.last_error,
                "transport": "serial",
            }


class Reader(threading.Thread):
    def __init__(self, port: str, baud: int, rate_hz: float, state: AttitudeState,
                 log_path: Optional[str]) -> None:
        super().__init__(name="fc-attitude-reader", daemon=True)
        self.cfg = FcConfig(port=port, baud=baud)
        self.period = 1.0 / max(1.0, float(rate_hz))
        self.state = state
        self.stop_event = threading.Event()
        self.link = None
        self.log_path = log_path
        self._csv = None
        self._writer = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._csv = open(log_path, "w", newline="", encoding="ascii")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(["utc_iso", "monotonic_s", "roll_deg", "pitch_deg", "yaw_deg"])

    def _connect(self) -> bool:
        try:
            link = make_link(self.cfg)
            if link.connect():
                self.link = link
                return True
            self.state.error("connect failed: %s" % (getattr(link, "last_error", "") or "unknown"))
        except Exception as exc:  # noqa: BLE001
            self.state.error("connect exception: %s" % exc)
        return False

    def run(self) -> None:
        while not self.stop_event.is_set():
            if self.link is None and not self._connect():
                self.stop_event.wait(1.0)
                continue
            try:
                payload = self.link.query(MSP_CMD.GET_ATTITUDE, timeout=0.25)
                if payload:
                    rpy = decode_attitude(payload)
                    if rpy is not None:
                        self.state.update(rpy)
                        if self._writer is not None:
                            self._writer.writerow([
                                datetime.now(timezone.utc).isoformat(),
                                round(time.monotonic(), 4),
                                round(rpy[0], 3), round(rpy[1], 3), round(rpy[2], 3),
                            ])
                    else:
                        self.state.error("short MSP_ATTITUDE payload")
                else:
                    self.state.error("no MSP_ATTITUDE reply")
            except Exception as exc:  # noqa: BLE001
                self.state.error("serial: %s" % exc)
                try:
                    self.link.close()
                except Exception:
                    pass
                self.link = None
            self.stop_event.wait(self.period)

    def stop(self) -> None:
        self.stop_event.set()
        if self.link is not None:
            try:
                self.link.close()
            except Exception:
                pass
            self.link = None
        if self._csv is not None:
            try:
                self._csv.flush()
                self._csv.close()
            except Exception:
                pass
            self._csv = None


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BF_Anti FC Attitude</title>
<style>
body{background:#111;color:#ddd;font-family:ui-monospace,monospace;margin:0;padding:16px}
h1{font-size:18px;color:#8cf;margin:0 0 10px}
.wrap{display:flex;gap:20px;flex-wrap:wrap}
canvas{background:#000;border:1px solid #333;border-radius:8px}
.nums{min-width:260px}
.row{display:flex;justify-content:space-between;border-bottom:1px solid #222;padding:7px 0}
.val{color:#8f8;font-weight:bold}
.small{color:#777;font-size:12px}
button{background:#222;color:#ddd;border:1px solid #555;border-radius:4px;padding:6px 10px;margin-right:6px;cursor:pointer}
.err{color:#f66;white-space:pre-wrap;max-width:700px}
</style></head><body>
<h1>BF_Anti FC 姿态实时检查</h1>
<div class="small">只读 MSP_ATTITUDE；不写 RC、不解锁、不转电机。用鼠标/触摸板旋转飞控板看数值与地平仪。</div>
<div class="wrap" style="margin-top:12px">
  <canvas id="h" width="520" height="330"></canvas>
  <div class="nums">
    <div class="row"><span>ROLL</span><span class="val" id="roll">--</span></div>
    <div class="row"><span>PITCH</span><span class="val" id="pitch">--</span></div>
    <div class="row"><span>YAW</span><span class="val" id="yaw">--</span></div>
    <div class="row"><span>样本</span><span class="val" id="samples">--</span></div>
    <div class="row"><span>数据年龄</span><span class="val" id="age">--</span></div>
    <div class="row"><span>错误</span><span class="val" id="errors">--</span></div>
    <p><button id="zero">记录当前为零点</button><button id="clear">恢复原始角度</button></p>
    <div id="err" class="err"></div>
    <p class="small">约定：ROLL positive / negative、PITCH positive / negative、YAW 0..360。先记录零点，再缓慢转动每个轴。</p>
  </div>
</div>
<script>
const cv=document.getElementById('h'), ctx=cv.getContext('2d');
let zero={r:0,p:0,y:0}, useZero=false, last=null;
function draw(roll,pitch,yaw){
  const w=cv.width,h=cv.height,cx=w/2,cy=h/2+20;
  ctx.save(); ctx.clearRect(0,0,w,h);
  ctx.translate(cx,cy); ctx.rotate(roll*Math.PI/180);
  const off=Math.max(-160,Math.min(160,pitch*3.2));
  ctx.fillStyle='#3a78b8'; ctx.fillRect(-w,-h,2*w,h+off);
  ctx.fillStyle='#8a5a36'; ctx.fillRect(-w,off,2*w,h);
  ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(-w,off); ctx.lineTo(w,off); ctx.stroke();
  for(let d=-60;d<=60;d+=10){ if(d===0) continue; const y=off-d*3.2; ctx.strokeStyle='rgba(255,255,255,.35)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(-38,y); ctx.lineTo(38,y); ctx.stroke(); }
  ctx.restore();
  ctx.strokeStyle='#ff0'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(cx,cy,52,0,Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx-62,cy); ctx.lineTo(cx+62,cy); ctx.moveTo(cx,cy-62); ctx.lineTo(cx,cy+62); ctx.stroke();
  ctx.fillStyle='#8f8'; ctx.font='14px ui-monospace,monospace'; ctx.fillText('YAW '+yaw.toFixed(1)+'°',12,24);
}
async function tick(){
  try{
    const r=await fetch('/attitude.json',{cache:'no-store'}); const d=await r.json();
    let roll=d.roll_deg,pitch=d.pitch_deg,yaw=d.yaw_deg;
    if(useZero){roll-=zero.r;pitch-=zero.p;yaw=(yaw-zero.y+360)%360;}
    document.getElementById('roll').textContent=d.roll_deg.toFixed(1)+'°';
    document.getElementById('pitch').textContent=d.pitch_deg.toFixed(1)+'°';
    document.getElementById('yaw').textContent=d.yaw_deg.toFixed(1)+'°';
    document.getElementById('samples').textContent=d.samples;
    document.getElementById('age').textContent=(d.age_ms==null?'--':d.age_ms.toFixed(0)+' ms');
    document.getElementById('errors').textContent=d.errors;
    document.getElementById('err').textContent=d.last_error||'';
    draw(roll,pitch,yaw);
  }catch(e){document.getElementById('err').textContent=String(e);}
  setTimeout(tick,50);
}
document.getElementById('zero').onclick=()=>{ if(!last) return; zero={r:last.roll_deg,p:last.pitch_deg,y:last.yaw_deg}; useZero=true;};
document.getElementById('clear').onclick=()=>{useZero=false;};
async function tick2(){ try{const r=await fetch('/attitude.json',{cache:'no-store'}); last=await r.json();}catch(e){} setTimeout(tick2,100);} 
tick(); tick2();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    state: AttitudeState

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/attitude.json":
            body = json.dumps(self.state.snapshot(), separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--http-port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--log", default="build/fc/attitude_view.csv")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    state = AttitudeState()
    log_path = None if args.no_log else args.log
    reader = Reader(args.port, args.baud, args.rate, state, log_path)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.http_port), Handler)
    server.daemon_threads = True
    reader.start()
    print("FC attitude viewer: http://%s:%d/ (serial %s)" % (args.host, args.http_port, args.port), flush=True)
    if log_path:
        print("CSV log: %s" % os.path.abspath(log_path), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
