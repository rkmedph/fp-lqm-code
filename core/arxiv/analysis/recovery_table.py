"""Table of individual-parameter recovery (extended version, appendix E) from recovery2.json.

    python3 recovery_table.py [--root PATH]
"""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, load_json

import numpy as np

SRC = ROOT / "recovery2.json"
OUT = ROOT / "arxiv" / "analysis" / "recovery_table_rows.tex"
PARS = ("mu", "sigma^2", "kappa", "M")   # столбцы est, попадающие в таблицу
NPAR = len(PARS)                         # пятый столбец est --- alpha/beta, он идёт в tab:S4
LAB = {"base": r"$\kappa=0.2$", "near": r"$\kappa=0.45$"}

R = load_json(SRC)

# --- проверка структуры до вычислений -----------------------------------------
bad = []
for key, v in R.items():
    parts = key.split("|")
    if len(parts) != 4:
        bad.append(f"{key}: ожидались 4 поля через '|', получено {len(parts)}"); continue
    sc, des, n, Mf = parts
    if sc not in LAB:
        bad.append(f"{key}: неизвестный сценарий {sc!r}, известны {sorted(LAB)}")
    if not (des.startswith("K") and des[1:].isdigit()):
        bad.append(f"{key}: поле дизайна {des!r} не имеет вида K<число>")
    if not (n.startswith("n") and n[1:].isdigit()):
        bad.append(f"{key}: поле числа клеток {n!r} не имеет вида n<число>")
    if Mf not in ("Mfixed", "Mfree"):
        bad.append(f"{key}: режим {Mf!r} не Mfixed и не Mfree")
    if "est" not in v:
        bad.append(f"{key}: нет поля 'est'"); continue
    est = np.asarray(v["est"], dtype=float)
    if est.ndim != 2 or est.shape[1] < NPAR:
        bad.append(f"{key}: est имеет форму {est.shape}, ожидалось (реплики, >= {NPAR}) "
                   f"для параметров {PARS}")
    elif len({tuple(np.asarray(w['est']).shape[1:]) for w in R.values() if 'est' in w}) != 1:
        bad.append(f"{key}: число столбцов est различается между сценариями")
    elif not np.isfinite(est).all():
        bad.append(f"{key}: в est есть не-конечные значения")
if bad:
    raise SystemExit("recovery2.json: " + "\n  ".join([""] + bad))


def num(x, d):
    """Число для LaTeX: минус как математический знак, а не дефис."""
    s = f"{x:.{d}f}"
    return ("$-$" + s[1:]) if s.startswith("-") else s


rows = []
for key, v in R.items():
    sc, des, n, Mf = key.split("|")
    est = np.asarray(v["est"], dtype=float)
    q = np.percentile(est, [25, 50, 75], axis=0)

    def cell(j, d=2):
        if Mf == "Mfixed" and j == 3:
            return "fixed"
        # разделитель диапазона --- именно '--', он не должен пострадать от знака минус
        return f"{num(q[1, j], d)} ({num(q[0, j], d)}--{num(q[2, j], d)})"

    rows.append(f"{LAB[sc]} & {des[1:]} & {int(n[1:])} & {'free' if Mf == 'Mfree' else 'fixed'} & "
                f"{cell(0, 3)} & {cell(1)} & {cell(2, 3)} & {cell(3, 1)}\\\\")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(rows) + "\n", encoding="utf8")
print("written", OUT, f"({len(rows)} строк)")
print("\n".join(rows))
