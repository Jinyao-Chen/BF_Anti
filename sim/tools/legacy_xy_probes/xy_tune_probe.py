import os, sys, socket, struct, threading, time, types
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard

udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True).start()
link=offboard.MspLink()
def setcfg(n,v):
    link.send(offboard.mspv2_command(0x3010,(n+" = "+v).encode())); time.sleep(0.1)
def pos():
    s=link.query(0x3015,is_v2=True)
    if s and len(s)>=27: return struct.unpack_from("<3f",s,3), struct.unpack_from("<3f",s,15)
    return (0,0,0),(0,0,0)

for n,v in [("ap_xy_roll_invert","OFF"),("ap_xy_pitch_invert","OFF"),("ap_position_p","2"),
            ("ap_position_i","8"),("ap_position_d","80"),("ap_position_a","0"),("ap_position_f","2"),
            ("ap_position_cutoff","15"),("ap_max_angle","15"),("ap_max_velocity","400")]:
    setcfg(n,v)
print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
print("hover 12s:", flush=True)
t0=time.time()
while time.time()-t0<12:
    p,v=pos(); print(f"  t={time.time()-t0:4.1f} E={p[0]:+6.2f} N={p[1]:+6.2f} U={p[2]:5.2f} vE={v[0]:+5.2f} vN={v[1]:+5.2f}", flush=True)
    time.sleep(2.0)
p0,_=pos(); print(f"step +E 1m from ({p0[0]:.2f},{p0[1]:.2f})", flush=True)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), p0[0]+1.0, p0[1], 10.0)
t0=time.time()
while time.time()-t0<6:
    p,v=pos(); print(f"  t={time.time()-t0:4.1f} dE={p[0]-p0[0]:+6.2f} vE={v[0]:+5.2f}", flush=True)
    time.sleep(0.5)
rc.idle(); running[0]=False
