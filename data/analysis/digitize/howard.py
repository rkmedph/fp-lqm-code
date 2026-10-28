"""Digitise the 6 MV (magenta) series of Howard et al 2018 Int J Part Ther, Fig. 3 (TrueBeam, 6 MV, 6 Gy/min).
Page 5 rendered at 400 dpi. Calibration from detected tick marks (decades + minor ticks, major dose ticks)."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from digi import load, Axes, blobs, errbar

IMG = "/home/claude/rawdata/mv/h400-05.png"
P = {  # panel: cell, proton series, x ticks (pix, Gy), y ticks (pix, SF), plot box (r0, r1, c0, c1), legend box to exclude
 "a": ("CHO", "71 MeV run", [1270, 1430, 1592, 1754, 1915], [0, 2, 4, 6, 8], [734, 1057, 636, 580, 540, 508], [0.1, 0.01, .2, .3, .4, .5], (395, 1185, 1273, 2160), (395, 560, 1900, 2160)),
 "b": ("CHO", "160 MeV run", [2322, 2484, 2646, 2808, 3132], [0, 2, 4, 6, 10], [736, 1058, 638, 582, 542, 510], [0.1, 0.01, .2, .3, .4, .5], (397, 1185, 2325, 3210), (397, 560, 2950, 3210)),
 "c": ("A549", "71 MeV run", [1275, 1440, 1604, 1768, 2094], [0, 2, 4, 6, 10], [1700, 1580, 1511, 1460, 1424, 1392, 1366, 1342, 1818, 1906, 1976], [0.1, .2, .3, .4, .5, .6, .7, .8, .05, .03, .02], (1285, 2074, 1278, 2165), (1285, 1450, 1900, 2165)),
 "d": ("A549", "160 MeV run", [2318, 2485, 2650, 2816, 2982, 3148], [0, 2, 4, 6, 8, 10], [1698, 1578, 1509, 1459, 1420, 1388, 1362, 1340, 1818, 1906, 1976], [0.1, .2, .3, .4, .5, .6, .7, .8, .05, .03, .02], (1285, 2074, 2324, 3215), (1285, 1450, 2960, 3215)),
 "e": ("T98", "71 MeV run", [1286, 1450, 1616, 1780, 1944, 2109], [0, 2, 4, 6, 8, 10], [2526, 2858, 2426, 2368, 2328], [0.1, 0.01, .2, .3, .4], (2176, 2964, 1289, 2178), (2176, 2340, 1910, 2178)),
 "f": ("T98", "160 MeV run", [2332, 2498, 2661, 2826, 2990], [0, 2, 4, 6, 8], [2525, 2856, 2424, 2367, 2326], [0.1, 0.01, .2, .3, .4], (2176, 2964, 2336, 3224), (2176, 2340, 2960, 3224)),
}


def magenta(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (R > 170) & (B > 130) & (G < 150) & ((B - G) > 30) & ((R - G) > 60)


def run():
    im = load(IMG); mag = magenta(im); out = []
    for k, (cell, run_, xp, xv, yp, yv, (r0, r1, c0, c1), (l0, l1, lc0, lc1)) in P.items():
        A = Axes(xp, xv, yp, yv)
        m = np.zeros_like(mag); m[r0:r1, c0:c1] = mag[r0:r1, c0:c1]; m[l0:l1, lc0:lc1] = False
        bl = blobs(m, 5, min_area=60)
        pts = []
        for b in bl:
            top, bot = errbar(m, b["row"], b["col"], b["box"], halfwidth=1, gap=3)
            pts.append(dict(D=round(A.x(b["col"]), 3), SF=A.y(b["row"]), SF_hi=A.y(top), SF_lo=A.y(bot), area=b["area"]))
        pts.sort(key=lambda p: p["D"])
        out.append(dict(panel=k, cell=cell, run=run_, calib_resid_x_Gy=A.xres, calib_resid_y_log10=A.yres, points=pts))
        print(k, cell, run_, f"resid x={A.xres:.3f} Gy, y={A.yres:.4f} dex")
        for p in pts: print(f"   D={p['D']:5.2f}  SF={p['SF']:.4f}  bar [{p['SF_lo']:.4f}, {p['SF_hi']:.4f}]  area={p['area']}")
    json.dump(out, open("/home/claude/rawdata/analysis/digitize/howard_6MV.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    run()
