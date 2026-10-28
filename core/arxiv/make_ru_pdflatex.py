"""pdflatex (T2A) variant of the Russian extended version, for arXiv, which builds with pdflatex."""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, require
AX = ROOT / 'arxiv'
SRC = require(AX/'main_arxiv_ru.tex', what='вход (сначала запустите build_ru.py)')
OUT = AX / 'rupdf' / 'ms_ru.tex'
OUT.parent.mkdir(parents=True, exist_ok=True)
s = SRC.read_text(encoding='utf8')
a = "\\usepackage{fontspec}\n\\usepackage[russian]{babel}\n\\setmainfont{Liberation Serif}\n\\setsansfont{Liberation Sans}\n\\setmonofont{Liberation Mono}"
assert s.count(a) == 1
s = s.replace(a, "\\usepackage[T2A]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage[russian]{babel}")
s = s.replace("Компиляция: xelatex main_arxiv_ru.tex (дважды).", "Компиляция: pdflatex ms_ru.tex (дважды).")
s = s.replace('\u0301', '').replace(r'{\scriptsize\setlength{\tabcolsep}{2.4pt}', r'{\scriptsize\setlength{\tabcolsep}{2.1pt}')
s = s.replace(r'{\tiny\setlength{\tabcolsep}{2pt}', r'{\tiny\setlength{\tabcolsep}{1.2pt}')
i = s.index(r'\caption{\label{tab:S4}'); j = s.index(r'\centering\small', i)
s = s[:j] + r'\centering\footnotesize' + s[j + len(r'\centering\small'):]
OUT.write_text(s, encoding='utf8')
print('written', OUT)
