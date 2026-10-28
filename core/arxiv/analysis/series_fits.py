"""Refit the 20 survival series of section 5.9 (LQ and exact model) and store the parameters for the
extended-version figure of data and fits. Uses the same routines as predictive.py; chi2 is compared with
the stored in-sample results."""
import sys, json
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, data_root, require, load_json

import numpy as np
from multiprocessing import Pool

ANA = data_root() / "analysis"
require(ANA / "predictive.py", ANA / "predictive_results.json", what="входные данные анализа")
OUT = ROOT / "arxiv" / "analysis" / "series_fits.json"
sys.path.insert(0, str(ANA))
import predictive as P

def job(item):
    name, (D, y, s, gi, K) = item
    L = P.fit_lq(D, y, s, gi, K); E = P.fit_exact(D, y, s, gi, K)
    return dict(name=name, D=D.tolist(), y=y.tolist(), s=s.tolist(), g=gi.tolist(), K=K,
                LQ=dict(par=L["par"].tolist(), chi2=L["chi2"]),
                EX=dict(lam=E["lam"].tolist(), F=E["F"], k=E["k"], nc=E["nc"], M=E["M"], chi2=E["chi2"]))

if __name__ == "__main__":
    DS = P.datasets()
    with Pool() as pool: res = pool.map(job, list(DS.items()), chunksize=1)
    stored = load_json(ANA / "predictive_results.json", require_keys=("name", "task"))
    full = [r for r in stored if r["task"] == "full"]
    ref = {r["name"]: r for r in full}
    if len(ref) != len(full):
        from collections import Counter
        dup = [k for k, c in Counter(r["name"] for r in full).items() if c > 1]
        raise SystemExit(f"predictive_results.json: повторяющиеся имена серий: {dup}")
    missing = sorted({r["name"] for r in res} - set(ref))
    if missing:
        raise SystemExit("в predictive_results.json нет серий: " + ", ".join(missing))
    extra = sorted(set(ref) - {r["name"] for r in res})
    if extra:
        print("предупреждение: в архиве есть серии, которых нет среди пересчитанных:", ", ".join(extra))
    for r in res:
        q = ref[r["name"]]; print(f"{r['name']:45s} LQ {r['LQ']['chi2']:8.3f} ({q['LQ']['chi2']:8.3f})  EX {r['EX']['chi2']:8.3f} ({q['EX']['chi2']:8.3f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    print("written", OUT)
