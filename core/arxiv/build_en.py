"""Assemble the extended arXiv version (English) from the journal manuscript and the new appendices."""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, require
AX = ROOT / 'arxiv'
SRC = ROOT / 'main_iopjournal_v36_full.tex'
require(SRC, AX/'appA_en.tex', AX/'appD_en.tex', AX/'appE_en.tex', what='исходник сборки')
src = SRC.read_text(encoding='utf8')
def rep(s, a, b, n=1):
    c = s.count(a)
    assert c == n, (c, a[:80]); return s.replace(a, b)
# ---- preamble and title block
i = src.index(r'\begin{abstract}')
pre = r"""%% Extended version for arXiv (article class). Figures fig1.pdf ... fig8.pdf, figS1.pdf in the same directory.
\documentclass[11pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[font=small]{caption}
\usepackage[hidelinks]{hyperref}
%% Microtype without glyph expansion, plus extra stretch to avoid overfull lines.
\usepackage[expansion=false]{microtype}
\setlength{\emergencystretch}{2em}
\tolerance=1500
\hyphenpenalty=300
\exhyphenpenalty=100

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{remark}{Remark}

\newcommand{\dd}{\mathrm{d}}
\newcommand{\Dr}{\dot D}
\newcommand{\E}{\mathbb{E}}
\emergencystretch=3em

\title{Lethal-threshold crossing in a stochastic lesion--repair model: exact
linear-quadratic coefficients, the Fokker--Planck limit and parameter identifiability\\[4pt]
\large Extended version}
\author{Konstantin Ryazanov\\[2pt]
\small Department of Medical Physics, Kaliningrad Regional Oncology Centre,\\
\small Kaliningrad, Russian Federation\\[2pt]
\small \texttt{rkmedph@gmail.com}\\[2pt]
\small \href{https://orcid.org/0009-0003-8809-8494}{ORCID: 0009-0003-8809-8494}}
\date{}

\begin{document}
\maketitle

"""
body = src[i:]
body = rep(body, r'\end{abstract}', r"""\end{abstract}

\noindent\textbf{Keywords:} linear-quadratic model, master equation, Fokker--Planck equation,
first-passage time, parameter identifiability, DNA damage repair, dose-rate effect

\medskip
\noindent\textit{About this version.} This extended version contains the full text of the
article together with detailed derivations of the exact results (appendix~\ref{app:deriv}),
a detailed description of the computational methods (appendix~\ref{app:comp}) and the
supplementary results, including per-dataset tables, all robustness variants and a figure of
all survival series with the fitted curves (appendix~\ref{app:supp}).
""")
# ---- labels and pointers
body = rep(body, r'\section{Exact results}', r"""\section{Exact results}\label{sec:exact}

Detailed derivations of the statements in this section are given in
appendix~\ref{app:deriv}.
""")
body = rep(body, 'supplementary table S4', r'table~\ref{tab:S4} (individual parameters in table~\ref{tab:Srec})')
body = rep(body, 'supplementary table S2c', r'table~\ref{tab:S2c}')
body = rep(body, 'supplementary table S1.', r'table~\ref{tab:S1}.')
body = rep(body, '(supplementary table S2)', r'(tables~\ref{tab:S2a} and~\ref{tab:S2b})')
body = rep(body, r"""described there.""", r"""described there.""", body.count('described there.'))
# pointer from model comparison to the series figure
body = rep(body, r"""The pooled series are analysed as a test of condition (R3) in
section~\ref{sec:res-transfer}.""", r"""The pooled series are analysed as a test of condition (R3) in
section~\ref{sec:res-transfer}. All series with both fits are shown in
figure~\ref{fig:Sseries} (appendix~\ref{app:supp}).""")
body = rep(body, r"""Table~\ref{tab:transfer} summarizes the joint fits of section~\ref{sec:meth-transfer}.""",
 r"""Table~\ref{tab:transfer} summarizes the joint fits of section~\ref{sec:meth-transfer};
all robustness variants are listed in table~\ref{tab:Stransfer}.""")
# ---- back matter
def macro(s, name, title):
    if ('\\' + name + '{') not in s:   # блок отсутствует (например, \ack удалён) --- пропускаем
        return s
    j = s.index('\\' + name + '{'); k = j + len(name) + 2; depth = 1
    while depth:
        depth += {'{': 1, '}': -1}.get(s[k], 0); k += 1
    content = s[j + len(name) + 2:k - 1]
    return s[:j] + '\\section*{' + title + '}\n' + content + s[k:]
body = macro(body, 'ack', 'Acknowledgements')
body = macro(body, 'funding', 'Funding')
body = macro(body, 'roles', 'Author contributions')
body = macro(body, 'data', 'Data availability')
body = rep(body, r"""\data""", r"""\data""", 0) if False else body
# ссылка на приложение теперь стоит прямо в \data{} полной версии
# ---- appendices
body = rep(body, "\\end{pmatrix},\\qquad\n\\det P=-\\frac{m_0^3}{(\\mu-\\kappa)M^2}\\ne0,\n\\end{equation*}\n", "\\end{pmatrix},\n\\end{equation*}\nwith $\\det P=-m_0^3/[(\\mu-\\kappa)M^2]\\ne0$, ")
A = (AX/'appA_en.tex').read_text(encoding='utf8')
D = (AX/'appD_en.tex').read_text(encoding='utf8')
Esup = (AX/'appE_en.tex').read_text(encoding='utf8')
body = rep(body, '\\appendix\n', '\\appendix\n' + A + '\n')
body = rep(body, '%% ===========================================================================\n\\section*{References}',
           D + '\n' + Esup + '\n%% ===========================================================================\n\\section*{References}')
body = body.replace('\\section*{References}', '\\clearpage\n\\section*{References}\n\\sloppy')
(AX/'main_arxiv.tex').write_text(pre + body, encoding='utf8')
print('written', AX/'main_arxiv.tex', len(pre + body), 'символов')
