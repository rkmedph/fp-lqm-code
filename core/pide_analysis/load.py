import pandas as pd, numpy as np
def load(path="data/PIDE_database.xlsx"):
    d=pd.read_excel(path,header=0)
    d.columns=[c.strip() for c in d.columns]
    return d
def photons(d):
    key=["#Publication","#PhotonExp"]
    cols=key+["PublicationName","Cells","CellClass","CellOrigin","CellCycle","DNAcontent","PhotonRadiation","ax_paper","bx_paper","ax_fit","bx_fit"]
    p=d[cols].groupby(key,as_index=False).first()
    return p
