"""Parse Polgár et al 2022 (Sci Data 9:555) database v2 (STOREDB:DATASET1252) into one record per dataset."""
import pandas as pd, numpy as np
SRC = "/home/claude/rawdata/polgar/database_v2.xlsx"
COLS = ['art','D','S','Smin','Smax','e1','fit','a','a_sd','a_lo','a_hi','b','b_sd','b_lo','b_hi','ab','as','as_sd','as_lo','as_hi',
        'Dc','Dc_sd','Dc_lo','Dc_hi','e2','cell','rad']

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan

def load(path=SRC):
    d = pd.read_excel(path, header=0); d.columns = COLS
    d["blk"] = d["D"].isna().cumsum(); d = d[d["D"].notna()]
    out = []
    for _, x in d.groupby("blk"):
        x = x.reset_index(drop=True)
        art = [str(v) for v in x["art"].dropna()]; cell = [str(v) for v in x["cell"].dropna()]; rad = [str(v) for v in x["rad"].dropna()]
        rec = dict(id=int(float(art[0])), authors=art[1] if len(art) > 1 else "", title=art[2] if len(art) > 2 else "",
                   year=art[3] if len(art) > 3 else "", doi=art[4] if len(art) > 4 else "", figure=art[5] if len(art) > 5 else "",
                   cell=cell[0] if cell else "", cell_info="; ".join(cell[1:]), radiation=rad[0] if rad else "", radiation_info="; ".join(rad[1:]),
                   D=x["D"].astype(float).to_numpy(), S=x["S"].astype(float).to_numpy(),
                   Smin=x["Smin"].map(num).to_numpy(), Smax=x["Smax"].map(num).to_numpy())
        fits = {}
        for _, r in x[x["fit"].notna()].iterrows():
            fits[str(r["fit"]).strip()] = {k: num(r[k]) for k in ("a", "a_sd", "b", "b_sd", "ab", "as", "as_sd", "Dc", "Dc_sd")}
        rec["fits_in_original"] = fits
        out.append(rec)
    return out

if __name__ == "__main__":
    R = load(); print(len(R), "datasets")
    from collections import Counter
    print(Counter(r["radiation"] for r in R))
    print("no whiskers:", sum(np.isnan(r["Smin"]).all() for r in R))
    print("cells:", Counter(r["cell"] for r in R).most_common(40))
    print("npts:", Counter(len(r["D"]) for r in R)); print("max dose dist", np.percentile([r["D"].max() for r in R],[0,25,50,75,100]))
    print("min dose", np.percentile([r["D"].min() for r in R],[0,50,100]))
