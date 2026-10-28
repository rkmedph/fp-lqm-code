"""Small helpers for digitising survival-curve figures from raster images.
Calibration from detected tick marks; markers from colour/darkness masks with morphological opening
(removes fit lines and error bars), error bars from the vertical extent of the mask at the marker column."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def tick_positions(mask, axis, pos, side, lo, hi, minlen=3, maxlen=40):
    """Positions of tick marks attached to an axis line.
    axis='x': horizontal axis at row pos, scan columns lo..hi, ticks extend up (side=-1) or down (+1).
    axis='y': vertical axis at column pos, scan rows lo..hi, ticks extend left (-1) or right (+1).
    Returns list of (position, length)."""
    out = []
    for k in range(lo, hi):
        L = 0
        for d in range(2, maxlen):
            r, c = (pos + side*d, k) if axis == "x" else (k, pos + side*d)
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]: L = d
            else: break
        if L >= minlen: out.append((k, L))
    # merge adjacent
    merged = []
    for k, L in out:
        if merged and k - merged[-1][-1][0] <= 1: merged[-1].append((k, L))
        else: merged.append([(k, L)])
    return [(float(np.mean([k for k, _ in g])), max(L for _, L in g)) for g in merged]


def blobs(mask, open_radius, min_area=10):
    """Connected components after opening with a disk; returns centroids (row, col), areas, bounding boxes."""
    r = open_radius; yy, xx = np.mgrid[-r:r+1, -r:r+1]; se = (xx**2 + yy**2) <= r*r
    m = ndi.binary_opening(mask, structure=se)
    lab, n = ndi.label(m)
    res = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        area = int((lab[sl] == i).sum())
        if area < min_area: continue
        cy, cx = ndi.center_of_mass(lab == i)
        res.append(dict(row=cy, col=cx, area=area, box=(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)))
    return res


def errbar(mask, row, col, box, halfwidth=1, gap=2):
    """Vertical extent of mask pixels contiguous with the marker in its central column(s): (top_row, bottom_row)."""
    c0, c1 = int(round(col)) - halfwidth, int(round(col)) + halfwidth + 1
    colmask = mask[:, c0:c1].any(1)
    top = box[0]; bot = box[1] - 1
    g = 0; r = top
    while r > 0:
        r -= 1
        if colmask[r]: top = r; g = 0
        else:
            g += 1
            if g > gap: break
    g = 0; r = bot
    while r < mask.shape[0] - 1:
        r += 1
        if colmask[r]: bot = r; g = 0
        else:
            g += 1
            if g > gap: break
    return top, bot


class Axes:
    def __init__(self, xpix, xval, ypix, yval, ylog=True):
        self.ax, self.bx = np.polyfit(xpix, xval, 1)
        yv = np.log10(yval) if ylog else np.asarray(yval, float)
        self.ay, self.by = np.polyfit(ypix, yv, 1); self.ylog = ylog
        self.xres = np.max(np.abs(np.polyval([self.ax, self.bx], xpix) - xval))
        self.yres = np.max(np.abs(np.polyval([self.ay, self.by], ypix) - yv))

    def x(self, col): return self.ax*col + self.bx
    def y(self, row):
        v = self.ay*row + self.by
        return 10**v if self.ylog else v


def marker_at(mask, col, row_lo, row_hi, halfw=4, minfill=5, avoid_rows=()):
    """Marker centre near a known column: rows where >= minfill mask pixels lie within col±halfw;
    returns centroid row of the longest contiguous run (filled marker body) and run length."""
    c0, c1 = int(round(col)) - halfw, int(round(col)) + halfw + 1
    cnt = mask[row_lo:row_hi, c0:c1].sum(1)
    good = cnt >= minfill
    runs, cur = [], []
    for i, gd in enumerate(good):
        if gd: cur.append(i)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    if not runs: return None
    best = max(runs, key=lambda r: (len(r), cnt[r].sum()))
    w = cnt[best].astype(float)
    return row_lo + float(np.sum(np.array(best)*w)/w.sum()), len(best)
