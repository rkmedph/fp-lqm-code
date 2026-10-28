"""Digitise Verbakel WFAR et al 2013 Acta Oncol 52:652-7, Fig. 3: SW1573 (A), D384 (B), T98 (C);
open squares = 6 MV FF (600 MU/min, 5.86 Gy/min), closed squares = 10 MV FFF (2400 MU/min, ~24 Gy/min); Varian TrueBeam;
SEM, n = 3. Template matching (filled vs hollow square) at nominal doses 2-12 Gy."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from digi import load, Axes

IMG = "/home/claude/rawdata/mv2/acta_f-002.png"
PANELS = {"SW1573": ([79.5, 135.5, 191.5, 247, 303], [0, 2, 4, 6, 8], [8.5, 119, 229, 339.5], (10, 338), (80, 425)),
          "D384": ([539.5, 594.5, 650.5, 705.5, 761.5, 817.5, 872.5], [0, 2, 4, 6, 8, 10, 12], [8, 118.5, 228.5, 339.5], (10, 338), (540, 882)),
          "T98": ([343, 398.5, 454.5, 510, 566], [0, 2, 4, 6, 8], [445, 555.5, 665.5, 776], (447, 774), (343, 685))}


def run():
    im = load(IMG); dark = im.mean(2) < 150
    yy, xx = np.mgrid[-6:7, -6:7]
    full = (np.abs(xx) <= 3) & (np.abs(yy) <= 3)
    outl = ((np.abs(xx) == 4) & (np.abs(yy) <= 4)) | ((np.abs(yy) == 4) & (np.abs(xx) <= 4)); inner = (np.abs(xx) <= 2) & (np.abs(yy) <= 2)
    out = []
    for cell, (xp, xv, yp, (r0, r1), (c0, c1)) in PANELS.items():
        A = Axes(xp, xv, yp, [1, .1, .01, .001])
        closed = {}
        for mode in ("10 MV FFF (closed)", "6 MV FF (open)"):
            pts = []
            for d in (2, 4, 6, 8, 10, 12):
                c = (d - A.bx)/A.ax
                if c > c1 - 6: c = c1 - 7          # 12 Gy marker sits on the right frame
                best = None; rlo, rhi = r0 + 6, r1 - 6
                if mode.startswith("6"):             # search near the closed marker (+-0.35 dex)
                    rc = closed[d]; dr = int(0.35/abs(A.ay)); rlo, rhi = max(rlo, rc - dr), min(rhi, rc + dr)
                for cc in range(int(c) - 4, int(c) + 5):
                    for rr in range(rlo, rhi):
                        w = dark[rr-6:rr+7, cc-6:cc+7]
                        if w.shape != full.shape: continue
                        if mode.startswith("10"):
                            s = w[full].mean()
                        else:
                            if w[outl].mean() < 0.7 or w[inner].mean() > 0.3: continue
                            s = w[outl].mean() - w[inner].mean()
                        if best is None or s > best[0]: best = (s, rr, cc)
                if best is None:                     # open marker hidden under the closed one
                    rr = closed[d]; s = float("nan"); cc = c; how = "coincident with closed marker (hidden)"
                else:
                    s, rr, cc = best; how = "template match"
                if mode.startswith("10") and best is not None:   # refine: centroid of the filled blob containing the match
                    from scipy import ndimage as ndi
                    sub = dark[rr-12:rr+13, int(cc)-12:int(cc)+13]
                    lab, n = ndi.label(ndi.binary_opening(sub, structure=np.ones((5, 5))))
                    k = lab[12, 12]
                    if k: cy, cx = ndi.center_of_mass(lab == k); rr = int(round(rr - 12 + cy))
                if mode.startswith("10"): closed[d] = rr
                pts.append(dict(D=float(d), SF=float(A.y(rr)), score=round(float(s), 2), dcol=round(cc - c, 1), how=how))
            out.append(dict(cell=cell, beam=mode, points=pts))
            print(cell, mode, [(p["D"], round(p["SF"], 4), p["score"], p["dcol"]) for p in pts])
    json.dump(out, open("/home/claude/rawdata/analysis/digitize/verbakel_raw.json", "w"), indent=1)


if __name__ == "__main__":
    run()
