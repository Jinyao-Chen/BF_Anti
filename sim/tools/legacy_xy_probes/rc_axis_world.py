import os, sys, socket, struct, threading, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazebo_target_bridge import GazeboPoseSource
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard

udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
th=threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True); th.start()
link=offboard.MspLink(); src=GazeboPoseSource("betaflight_world","iris_with_Betaflight","target_drone")
print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
running[0]=False; th.join(timeout=1.0)

ch=[1500]*16; ch[2]=1545; ch[4]=2000; ch[5]=2000; ch[7]=2000
running[0]=True
def sender():
    while running[0]:
        udp.sendto(struct.pack("<d"+"H"*16, time.time(), *ch), (offboard.UDP_IP, offboard.UDP_PORT)); time.sleep(0.02)
threading.Thread(target=sender,daemon=True).start()
offboard.send_target(link, type("S",(),{"cruise":1.0,"timeout_ms":0})(), 0.0,0.0,10.0)
time.sleep(9.0)
def att():
    p=link.query(108,is_v2=False)
    if p and len(p)>=6:
        r,pi,y=struct.unpack_from("<3h",p,0); return (r/10.0,pi/10.0,y/10.0)
    return (float('nan'),)*3
def truth_v():
    g=src.pose(); return np.array(g.own_velocity_enu) if g else None
def segment(label, axis, val, dur=2.0):
    ch[0]=1500; ch[1]=1500
    if axis=="roll": ch[0]=val
    if axis=="pitch": ch[1]=val
    time.sleep(0.3)
    v0=truth_v(); t0=time.time()
    vs=[]
    while time.time()-t0<dur:
        v=truth_v()
        if v is not None: vs.append(v)
        time.sleep(0.1)
    v1=truth_v(); a1=att()
    ch[0]=1500; ch[1]=1500; time.sleep(1.0)
    if v0 is None or v1 is None: print(label,"no truth"); return
    dv=v1-v0; dt=dur
    acc=dv/dt
    az=math.degrees(math.atan2(acc[0],acc[1]))%360.0 if math.hypot(acc[0],acc[1])>0.15 else float('nan')
    print(f"{label}: dv=({dv[0]:+5.2f},{dv[1]:+5.2f}) acc=({acc[0]:+5.2f},{acc[1]:+5.2f}) az={az:6.1f} deg  fw_att=({a1[0]:5.1f},{a1[1]:5.1f},{a1[2]:5.1f})", flush=True)
print("hover att=", tuple(round(v,1) for v in att()), flush=True)
segment("RC_roll=+100","roll",1600)
segment("RC_pitch=+100","pitch",1600)
running[0]=False; ch[2]=1000; ch[4]=1000; time.sleep(0.3); udp.close()
