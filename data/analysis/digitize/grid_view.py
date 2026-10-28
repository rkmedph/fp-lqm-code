"""Gridded zoom of a calibrated panel for manual read-off (WebPlotDigitizer-style)."""
import sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from digi import Axes

def grid(img, A, box, doses, out, zoom=4, step_dex=0.1):
    r0, r1, c0, c1 = box
    im = Image.open(img).convert("RGB").crop((c0, r0, c1, r1))
    im = im.resize(((c1-c0)*zoom, (r1-r0)*zoom), Image.LANCZOS); d = ImageDraw.Draw(im)
    lo, hi = np.log10(A.y(r1)), np.log10(A.y(r0))
    for v in np.arange(np.floor(lo/step_dex)*step_dex, hi + 1e-9, step_dex):
        row = (v - A.by)/A.ay; y = (row - r0)*zoom
        major = abs(v - round(v)) < 1e-6
        d.line([(0, y), (im.size[0], y)], fill=(255, 0, 0) if major else (255, 170, 170), width=1)
        d.text((2, y - 10), f"{v:.1f}", fill=(200, 0, 0))
    for D in doses:
        x = ((D - A.bx)/A.ax - c0)*zoom
        d.line([(x, 0), (x, im.size[1])], fill=(0, 140, 255), width=1)
    im.save(out)
