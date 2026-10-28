"""Сборка приложения E (дополнительные результаты) расширенной версии, английский.

    python3 make_appE.py [--root PATH] [--data PATH]
"""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, data_root, require

import re, csv

SUP = require(ROOT / "sub11" / "supplementary.tex", what="исходная supplementary.tex")
CSV = require(data_root() / "analysis" / "table_transfer.csv", what="таблица переносов")
REC = require(ROOT / "arxiv" / "analysis" / "recovery_table_rows.tex",
              what="строки таблицы Srec (сначала запустите recovery_table.py)")
OUT = ROOT / "arxiv" / "appE_en.tex"
sup = SUP.read_text(encoding="utf8")
def between(a, b):
    i = sup.find(a)
    if i < 0:
        raise SystemExit(
            f"{SUP}: не найден начальный маркер {a!r}.\n"
            "Этот скрипт рассчитан на ИСХОДНУЮ supplementary.tex (до того, как она стала\n"
            "собираться из appE_en.tex скриптом short/build_supp_en.py). Сейчас цепочка\n"
            "appE_en.tex -> supplementary.tex замкнута, и повторный прогон невозможен:\n"
            "правьте arxiv/appE_en.tex напрямую, он и есть источник.")
    j = sup.find(b, i)
    if j < 0:
        raise SystemExit(f"{SUP}: не найден конечный маркер {b!r} после позиции {i}")
    return sup[i:j]
# S1 longtable
s1 = between(r"{\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\section*{S2.").replace(r"{\footnotesize\setlength{\tabcolsep}{3.5pt}", r"{\scriptsize\setlength{\tabcolsep}{2.4pt}")
s1_intro = between(r"Datasets of the Polg", r"{\footnotesize")
s1 = s1.replace(r"\begin{longtable}{@{}rlllrcccccc@{}}", r"\begin{longtable}{@{}rlllrcccccc@{}}" + "\n" +
    r"\caption{Monotonicity test for the 91 low-dose photon datasets (section~\ref{sec:res-iso}). " + s1_intro.strip().replace("\n", " ") + r"}\label{tab:S1}\\")
s2_text = between(r"\textit{Least favourable configuration.}", r"\begin{table}[h]")
tabs = re.findall(r"\\begin\{table\}\[h\].*?\\end\{table\}", sup, re.S)
labs = {"Table S2a.": "tab:S2a", "Table S2b.": "tab:S2b", "Table S2c.": "tab:S2c", "Table S4.": "tab:S4"}
out_tabs = {}
for t in tabs:
    for key, lab in labs.items():
        if key in t:
            t2 = t.replace(r"\begin{table}[h]", r"\begin{table}[htbp]").replace(r"\caption*{" + key + " ", r"\caption{\label{" + lab + "}")
            t2 = t2.replace("(section 4.2;", r"(section~\ref{sec:meth-rec};")
            # колонка 4 содержит free/fixed, а не значение M
            t2 = t2.replace(r"cells & $M$ & true &", r"cells & mode & true &")
            t2 = t2.replace(r"cells & $M$ & $\hat\mu$", r"cells & mode & $\hat\mu$")
            # знак минус в числовых ячейках --- математический, а не дефис
            t2 = re.sub(r"(?<=[&\s])-(?=\d)", r"$-$", t2)
            out_tabs[lab] = t2
_missing = [k for k in labs.values() if k not in out_tabs]
if _missing:
    raise SystemExit(f"{SUP}: не найдены таблицы {_missing}; маркеры в файле изменились")
# transfer variants
rows = list(csv.DictReader(CSV.open(encoding='utf8')))
if not rows:
    raise SystemExit(f'{CSV}: нет строк')
vname = {"main": "main", "auto": "2nd digitization", "exclude": "excl.\\ ambiguous", "dig x2": "read-off $\\times2$", "dig x3": "read-off $\\times3$",
         "err+10%": "errors $+10\\%$", "err+20%": "errors $+20\\%$"}
def fmt_p(x):
    x = float(x); return "$<$0.001" if x < 0.001 else (f"{x:.3f}" if x < 0.1 else f"{x:.2f}")
def fmt_a(x):
    x = float(x); return f"{x:.1f}".replace("-", "$-$")
lines = []
for r in rows:
    v = vname[r["variant"]]
    if r["laboratory"] == "Li" and r["variant"] == "exclude": v = "excl.\\ 8\\,Gy"
    df = " / ".join(f"{float(x):.3f}" for x in r["dose factors (LQ)"].split()[1:])
    lr = " / ".join(f"{float(x):.3f}" for x in r["lambda_d ratios (exact)"].split()[1:])
    nc, M = [x.strip() for x in r["nc, M"].split(",")]
    lines.append(f"{r['laboratory']} {r['cell']} & {v} & {r['N']} & {df} & {lr} & {fmt_p(r['p_superposition(LQ rescaling vs independent)'])} & "
                 f"{fmt_a(r['dAIC exact-lambda minus LQ independent'])} & {fmt_a(r['dAIC exact-lambda minus LQ rescaling'])} & {fmt_p(r['p_GOF exact-lambda'])} & {nc}, {M}\\\\")
transfer = (r"""{\tiny\setlength{\tabcolsep}{2pt}
\begin{longtable}{@{}llrllcrrcc@{}}
\caption{Transfer of parameters between beams: all robustness variants (section~\ref{sec:meth-transfer}).
Dose factors $r_b$ of the dose-factor LQ fit and ratios of $\lambda_d$ of the exact model, both relative to
the first beam; $p_{\rm sup}$, $\Delta$AIC and $p_{\rm GOF}$ as in table~\ref{tab:transfer}; $(n_c,M)$ of the
shared-shape exact model.}\label{tab:Stransfer}\\
\toprule
Series & Variant & $N$ & $r_b$ (LQ) & $\lambda_d$ ratio & $p_{\rm sup}$ & $\Delta$AIC ind. & $\Delta$AIC resc. & $p_{\rm GOF}$ & $n_c$, $M$\\
\midrule
\endhead
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{longtable}}
""")
rec = REC.read_text(encoding='utf8')
rectab = r"""\begin{table}[htbp]
\caption{\label{tab:Srec}Recovery of the individual parameters in the computational experiment of
section~\ref{sec:meth-rec} (150 replicates per row): median and interquartile range. True values
$\mu=0.5$\,h$^{-1}$, $\sigma^2=1.5$\,h$^{-1}$, $M=4$ and $\kappa$ as indicated. $M$ at the upper bound
(200) indicates that the variance kinetics carry almost no information on $M$.}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrlllll}
\toprule
Scenario & $K$ & cells & mode & $\hat\mu$ (h$^{-1}$) & $\hat\sigma^2$ (h$^{-1}$) & $\hat\kappa$ (h$^{-1}$) & $\hat M$\\
\midrule
""" + rec + r"""\bottomrule
\end{tabular}
\end{table}
"""
body = r"""%% ===========================================================================
\section{Supplementary results}\label{app:supp}

This appendix contains the material that accompanies the journal version as supplementary
material, together with additional tables and a figure of all survival series with the fitted
curves.

\subsection{Monotonicity test: per-dataset results and calibration}\label{app:supp-iso}

Table~\ref{tab:S1} gives the $p$-values for every dataset and six of the 16 error models.
""" + s1 + "\n" + s2_text.strip() + "\n\n" + out_tabs["tab:S2a"] + "\n" + out_tabs["tab:S2b"] + "\n" + out_tabs["tab:S2c"] + r"""

\subsection{Parameter recovery from lesion kinetics}\label{app:supp-rec}

Table~\ref{tab:S4} summarizes the recovery of $\alpha/\beta$, and table~\ref{tab:Srec} that of
the individual parameters. With $M$ fixed, $\mu$ is recovered to within a few per cent in
all designs, whereas the estimates of $\sigma^2$ and $\kappa$ are almost perfectly
anticorrelated across replicates (correlation $-0.97$ to $-1.00$): at $10^3$ cells per point,
33--66\% of the fits with $M$ fixed put $\kappa$ at its lower bound 0 and compensate with a
larger $\sigma^2$. With $M$ free, the estimates of $M$ spread over the whole search interval.

""" + out_tabs["tab:S4"] + "\n" + rectab + r"""
\clearpage
\subsection{Transfer of parameters between beams: all variants}\label{app:supp-transfer}

""" + transfer + r"""
\subsection{Survival series and fitted curves}\label{app:supp-series}

Figure~\ref{fig:Sseries} shows all survival series of section~\ref{sec:res-pred} with the
LQ and exact-model fits used in table~\ref{tab:pred}. For series with several beams the
joint fits are shown (one $(\alpha,\beta)$ per beam for LQ; shared $(F_r,\kappa/\mu,n_c,M)$
and one $\lambda_d$ per beam for the exact model).

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{figS1.pdf}
\caption{\label{fig:Sseries}Survival series of table~\ref{tab:pred} (points with the error
bars used in the fits, converted to $S$) with the LQ fit (dashed) and the exact-model fit
(solid). Colours distinguish beams within a pooled series, in the order blue, red, green,
purple: Li 2024, 6\,MV, $^{192}$Ir, 225\,kVp, 50\,kVp; Steenken 2015, 6\,MV flattened, 6\,MV
FFF, 10\,MV flattened, 10\,MV FFF; Verbakel 2013, 6\,MV, 10\,MV FFF. Data: digitized or
tabulated from the cited publications.}
\end{figure}
"""

def firsthead(s, cont):
    import re
    def f(m):
        cap, hdr = m.group(1), m.group(2)
        return cap + hdr + "\\endfirsthead\n\\caption[]{" + cont + "}\\\\\n" + hdr + "\\endhead"
    return re.sub(r"(\\caption\{.*?\\label\{tab:S(?:1|transfer)\}\\\\\n)(.*?)\\endhead", f, s, flags=re.S)

body = firsthead(body, '(continued)')
OUT.write_text(body, encoding='utf8')
print('written', OUT, f'({len(lines)} строк переносов)')
