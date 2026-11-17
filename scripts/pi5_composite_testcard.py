#!/usr/bin/env python3
"""Draw a CVBS test card on /dev/fb0 (Pi 5 -> RP1 VEC -> J7 pad).

Geometry is read from sysfs, so this works for PAL 720x576, NTSC 720x480
and the VEC default 704x432i alike.
"""
import os, sys, time

FB = "/dev/fb0"

def fb_geometry():
    w, h = (int(x) for x in open("/sys/class/graphics/fb0/virtual_size").read().strip().split(","))
    bpp = int(open("/sys/class/graphics/fb0/bits_per_pixel").read().strip())
    return w, h, bpp // 8

def build(w, h):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    bars = [(235,235,235),(235,235,16),(16,235,235),(16,235,16),
            (235,16,235),(235,16,16),(16,16,235),(16,16,16)]
    bw = w // len(bars)
    for i, c in enumerate(bars):
        d.rectangle([i*bw, 0, (i+1)*bw - 1, int(h*0.62)], fill=c)
    d.rectangle([0, 0, w-1, h-1], outline=(255,255,255))
    cx, cy = w // 2, h // 2
    d.line([cx-40, cy, cx+40, cy], fill=(255,0,0), width=2)
    d.line([cx, cy-40, cx, cy+40], fill=(255,0,0), width=2)
    d.ellipse([cx-60, cy-60, cx+60, cy+60], outline=(255,255,0), width=2)
    d.rectangle([int(w*0.18), int(h*0.30), int(w*0.52), int(h*0.72)],
                outline=(0,255,0), width=3)
    font = small = None
    f1 = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    f2 = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(f1):
        font = ImageFont.truetype(f1, 30)
    if os.path.exists(f2):
        small = ImageFont.truetype(f2, 20)
    d.text((14, int(h*0.66)), "BF-ANTI  J7 CVBS %dx%d" % (w, h),
           fill=(255,255,255), font=font)
    d.text((14, int(h*0.66) + 40), "Pi5 RP1 VEC / %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
           fill=(210,210,210), font=small)
    return img.convert("RGBA").tobytes("raw", "BGRA")

def main():
    w, h, bpp = fb_geometry()
    data = build(w, h)
    with open(FB, "wb", buffering=0) as f:
        f.write(data)
    print("test card written: %dx%d %dbpp %d bytes" % (w, h, bpp*8, len(data)))

if __name__ == "__main__":
    main()
