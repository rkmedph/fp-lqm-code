"""Digitise Kalash et al 2014 In Vivo 28:1-12, Fig. 1 (3LL cells, TrueBeam 6/10 MV, 300-2400 MU/min, suspension irradiation).
Scanned colour figure: markers are hollow and washed-out; per dose column we cluster saturated pixels and assign
the cluster to the legend colour with the nearest mean RGB. Low-confidence digitisation."""
import sys, json
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from digi import load, Axes

IMG = "/home/claude/rawdata/mv/kalfig1-000.jpg"
A = Axes([122, 288, 454, 620, 786, 951], [0, 2, 4, 6, 8, 10],
         [298, 336, 382, 437, 499, 576, 675, 815, 1056], [.9, .8, .7, .6, .5, .4, .3, .2, .1])
LEG = {"10 MV 300 MU/min": (217, 152, 168), "10 MV 2400 MU/min": (141, 185, 170),
       "6 MV 300 MU/min": (133, 162, 192), "6 MV 1400 MU/min": (155, 198, 218)}


def run():
    im = load(IMG); sat = (im.max(2) - im.min(2)) > 35
    sat[:, :128] = False; sat[:, 945:] = False; sat[:230] = False; sat[1048:] = False
    sat[230:400, 520:945] = False          # legend box
    out = []
    for d in range(1, 9):
        c = (d - A.bx)/A.ax; c0, c1 = int(c) - 14, int(c) + 15
        win = np.zeros_like(sat); win[:, c0:c1] = sat[:, c0:c1]
        lab, n = ndi.label(ndi.binary_closing(win, iterations=2))
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            m = lab == i; area = int(m.sum())
            if area < 40: continue
            h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
            if w < 10: continue                      # skip thin error bars / line fragments
            rows, cols = np.nonzero(m & win)
            col = im[rows, cols].mean(0)
            name = min(LEG, key=lambda k: np.sum((np.array(LEG[k]) - col)**2))
            dist = float(np.sqrt(np.sum((np.array(LEG[name]) - col)**2)))
            out.append(dict(D=d, SF=A.y(rows.mean()), series=name, rgb=col.round().tolist(), colour_dist=dist, h=h, w=w, area=area))
    for o in out: print(o["D"], f"{o['SF']:.3f}", o["series"], o["rgb"], round(o["colour_dist"]), o["h"], o["w"], o["area"])
    json.dump(dict(calib_resid_x=A.xres, calib_resid_y=A.yres, points=out), open("/home/claude/rawdata/analysis/digitize/kalash_raw.json", "w"), indent=1, default=float)
    print("calib resid", A.xres, A.yres)


if __name__ == "__main__":
    run()
