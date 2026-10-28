"""Final Verbakel 2013 point set: closed squares (10 MV FFF) from template match + blob centroid (verbakel.py);
open squares (6 MV FF) read manually from 3x zoom crops with calibrated guide lines (verb_*.png) where visible,
otherwise coincident with the closed marker (hidden under it)."""
import json
raw = json.load(open("/home/claude/rawdata/analysis/digitize/verbakel_raw.json"))
closed = {r["cell"]: {p["D"]: p["SF"] for p in r["points"]} for r in raw if r["beam"].startswith("10")}
OPEN = {  # manual read-off (None = hidden under closed marker)
 "SW1573": {2: None, 4: 0.58, 6: 0.30, 8: 0.107, 10: 0.061, 12: 0.0277},
 "D384": {2: 0.86, 4: None, 6: 0.224, 8: None, 10: 0.0161, 12: 0.0030},
 "T98": {2: None, 4: 0.351, 6: 0.159, 8: None, 10: 0.0070, 12: 0.00119},
}
out = []
for cell in ("SW1573", "D384", "T98"):
    out.append(dict(cell=cell, beam="10 MV FFF", dose_rate="~24 Gy/min (2400 MU/min)",
                    points=[dict(D=d, SF=round(s, 5), how="template match + blob centroid") for d, s in sorted(closed[cell].items())]))
    out.append(dict(cell=cell, beam="6 MV FF", dose_rate="5.86 Gy/min (600 MU/min)",
                    points=[dict(D=float(d), SF=round(closed[cell][float(d)] if v is None else v, 5),
                                 how="hidden under closed marker (set equal)" if v is None else "manual read-off, 3x zoom") for d, v in OPEN[cell].items()]))
json.dump(dict(source="Verbakel WFAR et al 2013 Acta Oncol 52:652-7, Fig. 3", doi="10.3109/0284186X.2012.737021", linac="Varian TrueBeam",
               errors="SEM, n = 3 (bars mostly smaller than markers)", curves=out), open("/home/claude/rawdata/analysis/digitize/verbakel_curves.json", "w"), indent=1)
for c in out: print(c["cell"], c["beam"], [(p["D"], p["SF"]) for p in c["points"]])
