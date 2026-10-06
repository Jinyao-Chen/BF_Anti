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
print("arm:", offboard.arm(link, rc, timeout=40), flush=True)
rc.start_spool(); time.sleep(0.5)
for label,P,D in [("P1_D0",1,0),("P1_D40",1,40),("P0_D0",0,0)]:
    for n,v in [("ap_xy_roll_invert","OFF"),("ap_xy_pitch_invert","OFF"),("ap_position_p",str(P)),("ap_position_i","0"),
                ("ap_position_d",str(D)),("ap_position_a","0"),("ap_position_f","0"),("ap_position_cutoff","15"),
                ("ap_max_angle","10"),("ap_max_velocity","150")]:
        setcfg(n,v)
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0)
    time.sleep(3.0)
    t0=time.time(); last=None
    while time.time()-t0<6:
        p,v=pos(); last=(p,v)
        time.sleep(1.0)
    p,v=last
    print(f"{label}: after 6s E={p[0]:+6.2f} N={p[1]:+6.2f} U={p[2]:5.2f} vE={v[0]:+5.2f} vN={v[1]:+5.2f}", flush=True)
    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, 10.0); time.sleep(2.0)
rc.idle(); running[0]=False
