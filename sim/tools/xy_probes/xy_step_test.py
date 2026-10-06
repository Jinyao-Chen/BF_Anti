import os, sys, socket, struct, threading, time, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazebo_target_bridge import GazeboPoseSource
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard

udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True).start()
link=offboard.MspLink(); src=GazeboPoseSource("betaflight_world","iris_with_Betaflight","target_drone")

def read_state():
    p=link.query(0x3016,is_v2=True); s=link.query(0x3015,is_v2=True)
    a=link.query(108,is_v2=False); g=link.query(106,is_v2=False)
    out={}
    if p and len(p)>=36:
        fl=struct.unpack_from("<12B",p,0); fv=struct.unpack_from("<6f",p,12)
        out['flags']=fl; out['navvel']=fv[:3]; out['ang']=fv[3:5]
    if s and len(s)>=27:
        out['pos']=struct.unpack_from("<3f",s,3); out['vel']=struct.unpack_from("<3f",s,15)
    if a and len(a)>=6:
        r,pi,y=struct.unpack_from("<3h",a,0); out['att']=(r/10.0,pi/10.0,y/10.0)
    if g and len(g)>=12:
        gs,gc=struct.unpack_from("<HH",g,8); out['gps']=(gs/100.0,gc/10.0)
    return out

print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
print("climb to (0,0,10)", flush=True)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
time.sleep(9.0)
print("before step pos/navVel:", np.round(read_state().get('pos',[0,0,0]),2).tolist(), flush=True)
print("step to (5,0,10)", flush=True)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 5.0, 0.0, 10.0)
t0=time.time()
while time.time()-t0<9:
    st=read_state(); g=src.pose()
    tr=(np.round(g.own_position_enu,2).tolist() if g else None)
    if st.get('pos'):
        print(f"t={time.time()-t0:4.1f} pos=({st['pos'][0]:6.2f},{st['pos'][1]:6.2f},{st['pos'][2]:5.2f}) vel=({st['vel'][0]:6.2f},{st['vel'][1]:6.2f}) "
              f"nav=({st['navvel'][0]:5.2f},{st['navvel'][1]:5.2f}) ang=({st['ang'][0]:5.1f},{st['ang'][1]:6.1f}) att={tuple(round(v,1) for v in st['att'])} "
              f"gps(gs,crse)={None if 'gps' not in st else (round(st['gps'][0],2),round(st['gps'][1],1))} truth={tr}", flush=True)
    time.sleep(1.0)
rc.idle(); running[0]=False
