"""Search windows: physically plausible SF band per dose (excludes legend box, panel label and n.s. text).
Digitise Steenken et al 2015 Strahlenther Onkol 191:590, Fig. 2 (FLAT circles vs FFF squares, 6 and 10 MV, 5 Gy/min).
Doses = actual measured doses of Table 1; marker centres by template matching (hollow ring / hollow square) on the
dark-pixel mask in a narrow column window around each known dose; calibration from detected tick marks."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from digi import load, Axes

IMG = "/home/claude/rawdata/mv2/st_f-001.png"
DOSES = {("6 MV", "FFF"): [1.88, 5.22, 7.51, 11.26], ("10 MV", "FFF"): [1.83, 5.07, 7.30, 10.94],
         ("6 MV", "FLAT"): [1.97, 5.46, 7.86, 11.79], ("10 MV", "FLAT"): [1.98, 5.50, 7.91, 11.87]}
DEC = [1, 1e-1, 1e-2, 1e-3, 1e-4]
PANELS = {  # (cell, energy): x major ticks (0..14 by 2), y decade ticks, row window
 ("U251", "6 MV"): ([89, 129, 168, 207.5, 247, 286, 326, 365], [67, 121.5, 176.5, 231.5, 285], (50, 284)),
 ("U251", "10 MV"): ([471, 510, 549.5, 589, 628, 668, 707, 746], [67, 123, 178, 233, 285], (50, 284)),
 ("U251-MGMT", "6 MV"): ([88, 128, 168, 208, 247, 287, 327, 366], [377.5, 432.5, 489, 542.5, 594.5], (359, 594)),
 ("U251-MGMT", "10 MV"): ([470, 510, 549, 589, 628, 668, 707, 746], [377.5, 432.5, 487.5, 542.5, 594.5], (359, 594)),
 ("U87", "6 MV"): ([89, 129, 168, 208, 247, 287, 326, 366], [678, 733, 788, 844.5, 897], (661, 896)),
 ("U87", "10 MV"): ([472, 511, 551, 590, 629, 669, 708, 747], [684, 737, 793, 848, 900], (666, 899)),
}


def templates():
    yy, xx = np.mgrid[-5:6, -5:6]; r = np.hypot(xx, yy)
    ring = (r >= 2.3) & (r <= 3.8); ring_in = r < 1.6
    sq = ((np.abs(xx) == 3) & (np.abs(yy) <= 3)) | ((np.abs(yy) == 3) & (np.abs(xx) <= 3)); sq_in = (np.abs(xx) <= 1) & (np.abs(yy) <= 1)
    return {"FLAT": (ring, ring_in), "FFF": (sq, sq_in)}


def run():
    im = load(IMG); dark = im.mean(2) < 150; T = templates(); out = []
    for (cell, en), (xt, yt, (r0, r1)) in PANELS.items():
        A = Axes(xt, list(range(0, 15, 2)), yt, DEC)
        for mode in ("FLAT", "FFF"):
            outl, inn = T[mode]; pts = []
            for d in DOSES[(en, mode)]:
                c = (d - A.bx)/A.ax; best = None
                lo, hi = (0.2, 1.5) if d < 3 else (0.03, 0.7) if d < 6.5 else (0.003, 0.2) if d < 9 else (1e-4, 8e-3)
                rlo = max(r0 + 6, int((np.log10(hi) - A.by)/A.ay)); rhi = min(r1 - 6, int((np.log10(lo) - A.by)/A.ay))
                for cc in range(int(c) - 3, int(c) + 4):
                    for rr in range(rlo, rhi + 1):
                        w = dark[rr-5:rr+6, cc-5:cc+6]
                        if w.shape != outl.shape: continue
                        s = w[outl].mean() - w[inn].mean()
                        if best is None or s > best[0]: best = (s, rr, cc)
                s, rr, cc = best
                pts.append(dict(D=d, SF=float(A.y(rr)), row=rr, col=cc, col_expected=round(c, 1), score=round(float(s), 2)))
            out.append(dict(cell=cell, energy=en, mode=mode, points=pts, calib_resid=(A.xres, A.yres)))
            print(cell, en, mode, [(p["D"], round(p["SF"], 4), p["score"], p["col"] - p["col_expected"]) for p in pts])
    json.dump(out, open("/home/claude/rawdata/analysis/digitize/steenken_raw.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    run()
