"""Assemble the Russian extended arXiv version from ru/main_ru.tex and the Russian appendices."""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, require
AX = ROOT / 'arxiv'
SRC = ROOT / 'ru' / 'main_ru_v36_full.tex'
require(SRC, AX/'appA_ru.tex', AX/'appD_ru.tex', AX/'appE_ru.tex', what='исходник сборки')
src = SRC.read_text(encoding='utf8')
def rep(s, a, b, n=1):
    c = s.count(a); assert c == n, (c, a[:80]); return s.replace(a, b)
i = src.index(r'\begin{abstract}')
pre = r"""%% Расширенная версия для arXiv (класс article). Компиляция: xelatex main_arxiv_ru.tex (дважды).
%% Рисунки fig1.pdf ... fig8.pdf, figS1.pdf в той же папке.
\documentclass[11pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{fontspec}
\usepackage[russian]{babel}
\setmainfont{Liberation Serif}
\setsansfont{Liberation Sans}
\setmonofont{Liberation Mono}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[font=small]{caption}
\usepackage[hidelinks]{hyperref}
%% Выключка: microtype без растяжения глифов (совместимо с xelatex/fontspec),
%% плюс запас на растяжку — убирает переполненные строки в русском тексте.
\usepackage[expansion=false]{microtype}
\setlength{\emergencystretch}{2em}
\tolerance=1500
\hyphenpenalty=300
\exhyphenpenalty=100

\newtheorem{theorem}{Теорема}
\newtheorem{proposition}{Предложение}
\newtheorem{corollary}{Следствие}
\newtheorem{remark}{Замечание}
\renewcommand{\proofname}{Доказательство}

\newcommand{\dd}{\mathrm{d}}
\newcommand{\Dr}{\dot D}
\newcommand{\E}{\mathbb{E}}
\emergencystretch=3em

\title{Пересечение летального порога в стохастической модели индукции и репарации
повреждений: точные линейно-квадратичные коэффициенты, предел Фоккера--Планка и
идентифицируемость параметров\\[4pt]
\large Расширенная версия}
\author{Константин Рязанов\\[2pt]
\small Отделение медицинской физики, Онкологический центр Калининградской области,\\
\small Калининград, Российская Федерация\\[2pt]
\small \texttt{rkmedph@gmail.com}\\[2pt]
\small \href{https://orcid.org/0009-0003-8809-8494}{ORCID: 0009-0003-8809-8494}}
\date{}

\begin{document}
\maketitle

"""
body = src[i:]
body = rep(body, r'\end{abstract}', r"""\end{abstract}

\noindent\textbf{Ключевые слова:} линейно-квадратичная модель, управляющее уравнение (master equation),
уравнение Фоккера--Планка, время первого достижения, идентифицируемость параметров,
репарация повреждений ДНК, эффект мощности дозы

\medskip
\noindent\textit{Об этой версии.} Расширенная версия содержит полный текст статьи, а также
подробные выводы точных результатов (приложение~\ref{app:deriv}), подробное описание
вычислительных методов (приложение~\ref{app:comp}) и дополнительные результаты, включая
таблицы по отдельным наборам данных, все варианты проверки устойчивости и рисунок всех серий
выживаемости с подогнанными кривыми (приложение~\ref{app:supp}).
""")
body = rep(body, r'\section{Точные результаты}', r"""\section{Точные результаты}\label{sec:exact}

Подробные выводы утверждений этого раздела приведены в приложении~\ref{app:deriv}.
""")
body = rep(body, 'приведены в дополнительной таблице S4.', r'приведены в таблице~\ref{tab:S4} (отдельные параметры --- в таблице~\ref{tab:Srec}).')
body = rep(body, 'дополнительная таблица S2c)', r'таблица~\ref{tab:S2c})')
body = rep(body, 'приведены в дополнительной таблице S1.', r'приведены в таблице~\ref{tab:S1}.')
body = rep(body, '(дополнительная таблица S2)', r'(таблицы~\ref{tab:S2a} и~\ref{tab:S2b})')
body = rep(body, r'Объединённые серии анализируются как проверка условия (R3) в разделе~\ref{sec:res-transfer}.',
 r"""Объединённые серии анализируются как проверка условия (R3) в разделе~\ref{sec:res-transfer}.
Все серии с обеими подгонками показаны на рисунке~\ref{fig:Sseries} (приложение~\ref{app:supp}).""")
body = rep(body, r'Таблица~\ref{tab:transfer} обобщает совместные подгонки раздела~\ref{sec:meth-transfer}.',
 r"""Таблица~\ref{tab:transfer} обобщает совместные подгонки раздела~\ref{sec:meth-transfer}; все
варианты проверки устойчивости приведены в таблице~\ref{tab:Stransfer}.""")
def macro(s, name, title):
    if ('\\' + name + '{') not in s:   # блок отсутствует (например, \ack удалён) --- пропускаем
        return s
    j = s.index('\\' + name + '{'); k = j + len(name) + 2; depth = 1
    while depth:
        depth += {'{': 1, '}': -1}.get(s[k], 0); k += 1
    return s[:j] + '\\section*{' + title + '}\n' + s[j + len(name) + 2:k - 1] + s[k:]
body = macro(body, 'ack', 'Благодарности')
body = macro(body, 'funding', 'Финансирование')
body = macro(body, 'roles', 'Вклад автора')
body = macro(body, 'data', 'Доступность данных')
# ссылка на приложение теперь стоит прямо в \data{} полной версии
body = rep(body, "\\end{pmatrix},\\qquad\n\\det P=-\\frac{m_0^3}{(\\mu-\\kappa)M^2}\\ne0,\n\\end{equation*}\n", "\\end{pmatrix},\n\\end{equation*}\nгде $\\det P=-m_0^3/[(\\mu-\\kappa)M^2]\\ne0$, ")
A = (AX/'appA_ru.tex').read_text(encoding='utf8')
D = (AX/'appD_ru.tex').read_text(encoding='utf8')
Esup = (AX/'appE_ru.tex').read_text(encoding='utf8')
body = rep(body, '\\appendix\n', '\\appendix\n' + A + '\n')
body = rep(body, '%% ===========================================================================\n\\section*{Список литературы}',
           D + '\n' + Esup + '\n%% ===========================================================================\n\\section*{Список литературы}')
body = body.replace('\\section*{Список литературы}', '\\clearpage\n\\section*{Список литературы}\n\\sloppy')
(AX/'main_arxiv_ru.tex').write_text(pre + body, encoding='utf8')
print('written', AX/'main_arxiv_ru.tex', len(pre + body), 'символов')
