import os, sys, socket, struct, threading, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazebo_target_bridge import GazeboPoseSource
sys.path.insert(0, os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts"))
import offboard_position_example as offboard
udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); rc=offboard.RcLink(); running=[True]
threading.Thread(target=offboard.rc_thread,args=(udp,rc,running),daemon=True).start()
link=offboard.MspLink(); src=GazeboPoseSource("betaflight_world","iris_with_Betaflight","target_drone")
def setcfg(n,v):
    link.send(offboard.mspv2_command(0x3010,(n+" = "+v).encode())); time.sleep(0.05)
def fw():
    s=link.query(0x3015,is_v2=True)
    if s and len(s)>=27: return struct.unpack_from("<3f",s,3), struct.unpack_from("<3f",s,15)
    return (0,0,0),(0,0,0)
for n,v in [("ap_position_p","0"),("ap_position_i","0"),("ap_position_d","0"),("ap_position_a","0"),
            ("ap_position_f","0"),("ap_max_angle","15"),("ap_max_velocity","200")]:
    setcfg(n,v)
print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
time.sleep(9.0)
print("P=0 open hover; compare firmware vs truth velocity", flush=True)
t0=time.time()
while time.time()-t0<8:
    p,v=fw(); g=src.pose()
    tv=(g.own_velocity_enu if g else (0,0,0))
    print(f"t={time.time()-t0:4.1f} fw_pos=({p[0]:+6.2f},{p[1]:+6.2f}) fw_vel=({v[0]:+6.2f},{v[1]:+6.2f}) truth_vel=({tv[0]:+6.2f},{tv[1]:+6.2f})", flush=True)
    time.sleep(0.15)
rc.idle(); running[0]=False
