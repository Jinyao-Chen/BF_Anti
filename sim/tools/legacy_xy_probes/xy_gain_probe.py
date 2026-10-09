import os, sys, socket, struct, threading, time, types
import numpy as np
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard

udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True).start()
link=offboard.MspLink()

def setcfg(name,val):
    link.send(offboard.mspv2_command(0x3010, (name+" = "+val).encode())); time.sleep(0.12)

def pos():
    s=link.query(0x3015,is_v2=True)
    if s and len(s)>=27: return struct.unpack_from("<3f",s,3), struct.unpack_from("<3f",s,15)
    return (0,0,0),(0,0,0)

print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
time.sleep(9.0)
print("hover:", np.round(pos()[0],2).tolist(), flush=True)

for label,gains in [("upstream-def", dict(p=30,i=30,d=30,a=30,f=30,cut=30,maxang=50,maxv=500))]:
    setcfg("ap_position_p",str(gains['p'])); setcfg("ap_position_i",str(gains['i']))
    setcfg("ap_position_d",str(gains['d'])); setcfg("ap_position_a",str(gains['a']))
    setcfg("ap_position_f",str(gains['f'])); setcfg("ap_position_cutoff",str(gains['cut']))
    setcfg("ap_max_angle",str(gains['maxang'])); setcfg("ap_max_velocity",str(gains['maxv']))
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
    time.sleep(4.0)
    p0,_=pos()
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0+1.0, 0.0, 10.0)
    print(f"{label}: gains={gains}", flush=True)
    t0=time.time()
    while time.time()-t0<5.0:
        p,v=pos()
        print(f"   t={time.time()-t0:3.1f} dE={p[0]-0.0:+6.2f} dN={p[1]-0.0:+6.2f} vE={v[0]:+6.2f}", flush=True)
        time.sleep(0.5)
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
    time.sleep(3.0)
rc.idle(); running[0]=False
