"""Перевод приложения E на русский: читает arxiv/appE_en.tex, пишет arxiv/appE_ru.tex.

    python3 make_appE_ru.py [--root PATH]
"""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, require

import re

SRC = require(ROOT / "arxiv" / "appE_en.tex",
              what="английское приложение E (сначала запустите make_appE.py)")
OUT = ROOT / "arxiv" / "appE_ru.tex"
s = SRC.read_text(encoding="utf8")
def rep(a, b, n=1):
    global s
    c = s.count(a); assert c == n, (c, a[:70]); s = s.replace(a, b)
# prose: replace everything between markers
def block(start, end, new):
    global s
    i = s.index(start); j = s.index(end, i); s = s[:i] + new + s[j:]
rep(r"\section{Supplementary results}\label{app:supp}", r"\section{Дополнительные результаты}\label{app:supp}")
block("This appendix contains the material", r"\subsection{Monotonicity test", "В это приложение включены материалы, сопровождающие журнальную версию в качестве\nдополнительных, а также дополнительные таблицы и рисунок всех серий выживаемости с\nподогнанными кривыми.\n\n")
rep(r"\subsection{Monotonicity test: per-dataset results and calibration}", r"\subsection{Тест монотонности: результаты по наборам данных и калибровка}")
rep("Table~\\ref{tab:S1} gives the $p$-values for every dataset and six of the 16 error models.",
    "В таблице~\\ref{tab:S1} приведены $p$-значения для каждого набора данных и шести из 16 моделей ошибок.")
block(r"\caption{Monotonicity test for the 91", r"\label{tab:S1}", r"\caption{Тест монотонности для 91 низкодозового фотонного набора данных (раздел~\ref{sec:res-iso}). Наборы данных базы Polg\'ar \textit{et al} (2022), нумерация как в базе. $p$-значения теста неубывания $-\ln S/D$, калиброванного в наименее благоприятной конфигурации (10000 нулевых реплик на набор данных и модель ошибок); $k$ масштабирует приведённые бары, $\sigma_c$ --- стандартное отклонение общей ошибки нормировки. Жирным: $p<0{,}05$. Сами данные о выживаемости здесь не воспроизводятся; они доступны в базе.}")
rep("ID & Cells & Radiation & Source & $n$", "№ & Клетки & Излучение & Источник & $n$", 2)
block(r"\textit{Least favourable configuration.}", r"\begin{table}[htbp]", r"""\textit{Наименее благоприятная конфигурация.} Статистика --- квадрат расстояния в метрике
обратной ковариационной матрицы от наблюдаемого вектора $G_i=-\ln S_i/D_i$ до конуса
неубывающих векторов (смещение нормировки исключено с гауссовым штрафом, что по тождеству
Вудбери эквивалентно ковариации $\mathrm{diag}(\sigma_i^2/D_i^2)+\sigma_c^2vv^{\rm T}$,
$v_i=1/D_i$). Для истинного $G$ из конуса $\mathcal C$ имеем $\mathcal C-G\supseteq\mathcal C$,
так что статистика стохастически наибольшая при постоянном $G$; критические значения,
смоделированные там, справедливы для всей нулевой гипотезы при гауссовой модели ошибок.

""")
block(r"\caption{\label{tab:S2a}", r"\centering\small", r"""\caption{\label{tab:S2a}Отклонения при $p<0{,}05$ среди 91 набора данных: калибровка в наименее
благоприятной конфигурации (10000 реплик) / параметрический бутстреп от подогнанной кривой с
ограничением (600 реплик) / число решений, различающихся между двумя независимыми прогонами
наименее благоприятной калибровки (2000 и 10000 реплик).}
""")
block(r"\caption{\label{tab:S2b}", r"\centering\small", r"""\caption{\label{tab:S2b}Частота отклонений (\%) при номинальном уровне 5\% на смоделированных
выпуклых кривых на 91 дизайне (364 набора данных на ячейку, по четыре на дизайн): бутстреп /
наименее благоприятная калибровка. Истинные ошибки равны приведённым барам; модель нормировки
совпадает с предполагаемой тестом. На границе при 1820 наборах на ячейку наименее
благоприятный тест дал 5,5, 5,5, 5,1 и 4,5\% при $\sigma_c=0$, 3, 5 и 10\% (стандартная
ошибка Монте-Карло 0,5\%).}
""")
block(r"\caption{\label{tab:S2c}", r"\centering\small", r"""\caption{\label{tab:S2c}Частота отклонений (\%) при номинальном уровне 5\% на границе ($-\ln S$
линейна по дозе) для всех 16 моделей ошибок: истинные ошибки --- $k$ приведённых баров и
смещение нормировки $\ln c\sim N(0,\sigma_c^2)$, оба как предполагает тест; 2000 нулевых
реплик на дизайн, 1820 смоделированных наборов на ячейку (стандартная ошибка Монте-Карло
0,5\%).}
""")
rep(r"$-\ln S$ linear ($g''=0$) &", r"$-\ln S$ линейна ($g''=0$) &")
rep(r"LQ, $\beta\sim U(0,0.06)$\,Gy$^{-2}$ &", r"LQ, $\beta\sim U(0;\,0.06)$\,Гр$^{-2}$ &")
rep(r"LQ, $\beta=0.005$\,Gy$^{-2}$ &", r"LQ, $\beta=0.005$\,Гр$^{-2}$ &")
rep(r"linear to 1\,Gy, convex above &", r"линейна до 1\,Гр, выше выпукла &")
rep(r"True curve & $\sigma_c=0$", r"Истинная кривая & $\sigma_c=0$")
rep(r"\subsection{Parameter recovery from lesion kinetics}", r"\subsection{Восстановление параметров по кинетике повреждений}")
block("Table~\\ref{tab:S4} summarizes the recovery", r"\begin{table}[htbp]", r"""Таблица~\ref{tab:S4} обобщает восстановление $\alpha/\beta$, а таблица~\ref{tab:Srec} ---
отдельных параметров. При фиксированном $M$ параметр $\mu$ восстанавливается с точностью до
нескольких процентов во всех дизайнах, тогда как оценки $\sigma^2$ и $\kappa$ почти идеально
антикоррелированы между репликами (корреляция от $-0{,}97$ до $-1{,}00$): при $10^3$ клеток
на точку 33--66\% подгонок с фиксированным $M$ дают $\kappa$ на нижней границе 0 и
компенсируют это большим $\sigma^2$. При свободном $M$ его оценки разбросаны по всему
интервалу поиска.

""")
block(r"\caption{\label{tab:S4}", r"\centering\small", r"""\caption{\label{tab:S4}Восстановление $\alpha/\beta$ (Гр) по смоделированной кинетике повреждений
(раздел~\ref{sec:meth-rec}; 150 реплик на строку): истинное значение, медиана и
межквартильный размах оценки и доля подгонок (\%) с $\kappa$ или $M$ на границе интервала
поиска ($0\le\kappa\le2$\,ч$^{-1}$, $0{,}2\le M\le200$). Оценки вне допустимой области
($\sigma^2<\mu$) могут давать отрицательное $\alpha/\beta$. Колонка \emph{режим} указывает, оценивался ли $M$ совместно (своб.) или был закреплён на истинном значении (фикс.).}
""")
rep(r"Scenario & $K$ & cells & mode & true & median & IQR & $\kappa$ at bound & $M$ at bound",
    r"Сценарий & $K$ & клеток & режим & истина & медиана & МКР & $\kappa$ на границе & $M$ на границе")
block(r"\caption{\label{tab:Srec}", r"\centering\scriptsize", r"""\caption{\label{tab:Srec}Восстановление отдельных параметров в вычислительном эксперименте
раздела~\ref{sec:meth-rec} (150 реплик на строку): медиана и межквартильный размах. Истинные
значения $\mu=0{,}5$\,ч$^{-1}$, $\sigma^2=1{,}5$\,ч$^{-1}$, $M=4$ и $\kappa$, как указано. $M$
на верхней границе (200) означает, что кинетика дисперсии почти не несёт информации об $M$. Колонка \emph{режим} указывает, оценивался ли $M$ совместно (своб.) или был закреплён на истинном значении (фикс.).}
""")
rep(r"Scenario & $K$ & cells & mode & $\hat\mu$ (h$^{-1}$) & $\hat\sigma^2$ (h$^{-1}$) & $\hat\kappa$ (h$^{-1}$) & $\hat M$",
    r"Сценарий & $K$ & клеток & режим & $\hat\mu$ (ч$^{-1}$) & $\hat\sigma^2$ (ч$^{-1}$) & $\hat\kappa$ (ч$^{-1}$) & $\hat M$")
rep(r"\subsection{Transfer of parameters between beams: all variants}", r"\subsection{Перенос параметров между пучками: все варианты}")
block(r"\caption{Transfer of parameters between beams: all robustness", r"\label{tab:Stransfer}", r"""\caption{Перенос параметров между пучками: все варианты проверки устойчивости
(раздел~\ref{sec:meth-transfer}). Дозовые множители $r_b$ подгонки LQ с дозовым множителем и
отношения $\lambda_d$ точной модели, оба относительно первого пучка; $p_{\rm sup}$, $\Delta$AIC
и $p_{\rm GOF}$, как в таблице~\ref{tab:transfer}; $(n_c,M)$ точной модели с общей формой.}""")
rep(r"Series & Variant & $N$ & $r_b$ (LQ) & $\lambda_d$ ratio & $p_{\rm sup}$ & $\Delta$AIC ind. & $\Delta$AIC resc.",
    r"Серия & Вариант & $N$ & $r_b$ (LQ) & отношение $\lambda_d$ & $p_{\rm sup}$ & $\Delta$AIC незав. & $\Delta$AIC масшт.", 2)
for a, b in [("& main &", "& основной &"), ("& 2nd digitization &", "& 2-я оцифровка &"), ("& excl.\\ ambiguous &", "& без неоднозначных &"),
             ("& read-off $\\times2$ &", "& считывание $\\times2$ &"), ("& read-off $\\times3$ &", "& считывание $\\times3$ &"),
             ("& errors $+10\\%$ &", "& ошибки $+10\\%$ &"), ("& errors $+20\\%$ &", "& ошибки $+20\\%$ &"), ("& excl.\\ 8\\,Gy &", "& без 8\\,Гр &")]:
    s = s.replace(a, b)
rep(r"\subsection{Survival series and fitted curves}", r"\subsection{Серии выживаемости и подогнанные кривые}")
block("Figure~\\ref{fig:Sseries} shows all survival series", r"\begin{figure}[htbp]", r"""На рисунке~\ref{fig:Sseries} показаны все серии выживаемости раздела~\ref{sec:res-pred} с
подгонками LQ и точной модели, использованными в таблице~\ref{tab:pred}. Для серий с
несколькими пучками показаны совместные подгонки (по одной паре $(\alpha,\beta)$ на пучок для
LQ; общие $(F_r,\kappa/\mu,n_c,M)$ и по одному $\lambda_d$ на пучок для точной модели).

""")
block(r"\caption{\label{fig:Sseries}", r"\end{figure}", r"""\caption{\label{fig:Sseries}Серии выживаемости таблицы~\ref{tab:pred} (точки с барами ошибок,
использованными в подгонках, пересчитанными в $S$) с подгонкой LQ (штриховая линия) и
подгонкой точной модели (сплошная). Цвета различают пучки внутри объединённой серии в порядке синий,
красный, зелёный, фиолетовый: Li 2024 --- 6\,МВ, $^{192}$Ir, 225\,кВп, 50\,кВп; Steenken 2015 ---
6\,МВ с фильтром, 6\,МВ FFF, 10\,МВ с фильтром, 10\,МВ FFF; Verbakel 2013 --- 6\,МВ, 10\,МВ FFF.
Данные оцифрованы или взяты из таблиц цитируемых публикаций; надписи на рисунке на
английском.}
""")
s = s.replace(r"\includegraphics[width=\textwidth]{figS1.pdf}", r"\includegraphics[width=\textwidth]{figS1.pdf}")
# decimal commas in table rows and labels with fixed/free
out = []
for line in s.split("\n"):
    if "&" in line and "tabcolsep" not in line:
        line = re.sub(r"(?<=\d)\.(?=\d)", "{,}", line)
        line = line.replace("& fixed &", "& фикс. &").replace("& free &", "& своб. &").replace("& fixed\\\\", "& фикс.\\\\")
    out.append(line)
s = "\n".join(out)
s = s.replace("X-ray", "рентген").replace(r"60Co $\gamma$", r"$^{60}$Co $\gamma$").replace(r"137Cs $\gamma$", r"$^{137}$Cs $\gamma$")

def firsthead(s, cont):
    import re
    def f(m):
        cap, hdr = m.group(1), m.group(2)
        return cap + hdr + "\\endfirsthead\n\\caption[]{" + cont + "}\\\\\n" + hdr + "\\endhead"
    return re.sub(r"(\\caption\{.*?\\label\{tab:S(?:1|transfer)\}\\\\\n)(.*?)\\endhead", f, s, flags=re.S)

s = s.replace('{(continued)}', '{(продолжение)}')
OUT.write_text(s, encoding='utf8')
print('written', OUT)
left = [l for l in s.split("\n") if re.search(r"\b(the|of|and|with|per)\b", l) and "&" not in l]
print("possible untranslated lines:", len(left)); print("\n".join(left[:20]))
