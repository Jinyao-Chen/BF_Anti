import os, sys, socket, struct, threading, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazebo_target_bridge import GazeboPoseSource
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard

udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True).start()
link=offboard.MspLink(); src=GazeboPoseSource("betaflight_world","iris_with_Betaflight","target_drone")

def st():
    s=link.query(0x3015,is_v2=True); p=link.query(0x3016,is_v2=True); a=link.query(108,is_v2=False)
    o={}
    if s and len(s)>=27: o['pos']=struct.unpack_from("<3f",s,3); o['vel']=struct.unpack_from("<3f",s,15)
    if p and len(p)>=36:
        fv=struct.unpack_from("<6f",p,12); o['nav']=fv[:2]; o['ang']=fv[3:5]
    if a and len(a)>=6:
        r,pi,y=struct.unpack_from("<3h",a,0); o['att']=(r/10.0,pi/10.0,y/10.0)
    return o

print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
time.sleep(10.0)
base=st().get('pos',(0,0,0)); print(f"hover base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f})", flush=True)

def step(dx,dy,label):
    px,py,pz=base
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), px+dx, py+dy, 10.0)
    t0=time.time()
    while time.time()-t0<4.0:
        s=st()
        if s.get('pos'):
            print(f"  {label} t={time.time()-t0:3.1f} d=({s['pos'][0]-px:5.2f},{s['pos'][1]-py:5.2f}) v=({s['vel'][0]:5.2f},{s['vel'][1]:5.2f}) nav={s.get('nav')} ang={s.get('ang')} att={tuple(round(v,1) for v in s.get('att',(0,0,0)))}", flush=True)
        time.sleep(0.5)
    # return to base
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), px, py, 10.0)
    time.sleep(3.0)

step(1.0, 0.0, "E+1m")
step(0.0, 1.0, "N+1m")
rc.idle(); running[0]=False
