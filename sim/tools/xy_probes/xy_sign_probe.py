import os, sys, socket, struct, threading, time, types
import numpy as np
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard

udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True).start()
link=offboard.MspLink()

def setcfg(name,val):
    link.send(offboard.mspv2_command(0x3010, (name+" = "+val).encode()))
    time.sleep(0.15)

def pos():
    s=link.query(0x3015,is_v2=True)
    if s and len(s)>=27:
        return struct.unpack_from("<3f",s,3), struct.unpack_from("<3f",s,15)
    return (0,0,0),(0,0,0)

print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
time.sleep(9.0)
print("hover:", np.round(pos()[0],2).tolist(), flush=True)

for ri,pi in [(0,0),(1,0),(0,1),(1,1)]:
    setcfg("ap_xy_roll_invert", "ON" if ri else "OFF")
    setcfg("ap_xy_pitch_invert", "ON" if pi else "OFF")
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
    time.sleep(3.0)
    p0,_=pos()
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), p0[0]+1.0, p0[1], 10.0)
    time.sleep(1.8)
    p1,v1=pos()
    de=p1[0]-p0[0]; vne=v1[0]
    print(f"roll_invert={ri} pitch_invert={pi}: dE={de:+.2f} vE={vne:+.2f} (correct if +)", flush=True)
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
    time.sleep(2.5)
rc.idle(); running[0]=False
