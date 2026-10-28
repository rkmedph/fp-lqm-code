"""Raw survival points transcribed from the uploaded papers (see 'source' fields)."""
import numpy as np
D4 = [2, 4, 6, 8]
LI24 = {  # Li J et al 2024 Phys Med Biol 69 205008, appendix A.2: mean SF (± SEM)
 ("HCT116","6 MV"):   ([3.52e-1,1.05e-1,1.60e-2,3.0e-3],[2.9e-2,4.0e-3,1.0e-3,1.0e-3]),
 ("HCT116","192Ir"):  ([3.54e-1,8.1e-2,1.5e-2,2.0e-3],[2.0e-2,5.0e-3,1.0e-3,1.0e-3]),
 ("HCT116","225 kVp"):([3.67e-1,6.6e-2,7.0e-3,1.0e-3],[2.3e-2,6.0e-3,1.0e-3,1.0e-3]),
 ("HCT116","50 kVp"): ([1.98e-1,3.4e-2,3.0e-3,1.0e-3],[2.0e-2,2.0e-3,1.0e-3,1.0e-3]),
 ("HeLa","6 MV"):     ([5.15e-1,1.21e-1,3.4e-2,7.0e-3],[4.3e-2,3.5e-2,6.0e-3,1.0e-3]),
 ("HeLa","192Ir"):    ([5.32e-1,1.57e-1,2.4e-2,8.0e-3],[6.1e-2,2.2e-2,5.0e-3,2.0e-3]),
 ("HeLa","225 kVp"):  ([5.29e-1,1.23e-1,2.7e-2,4.0e-3],[2.6e-2,8.0e-3,3.0e-3,1.0e-3]),
 ("HeLa","50 kVp"):   ([4.13e-1,1.14e-1,1.7e-2,2.0e-3],[3.7e-2,1.7e-2,2.0e-3,1.0e-3]),
 ("PC3","6 MV"):      ([5.37e-1,1.94e-1,3.9e-2,5.0e-3],[5.9e-2,7.0e-2,2.4e-2,3.0e-3]),
 ("PC3","192Ir"):     ([5.29e-1,1.87e-1,3.8e-2,5.0e-3],[3.7e-2,2.1e-2,3.0e-3,1.0e-3]),
 ("PC3","225 kVp"):   ([4.04e-1,1.08e-1,1.8e-2,2.0e-3],[1.01e-1,2.6e-2,8.0e-3,1.0e-3]),
 ("PC3","50 kVp"):    ([3.89e-1,6.9e-2,1.1e-2,1.0e-3],[8.7e-2,1.0e-2,4.0e-3,1.0e-3]),
}
CURVES = []
for (cell, beam), (S, e) in LI24.items():
    CURVES.append(dict(cell=cell, beam=beam, D=np.array(D4, float), S=np.array(S), sS=np.array(e),
                       source="Li et al 2024 PMB, App. A.2 (SEM)"))
# Murakami et al 2004 Radiat Environ Biophys 43:91, Table 1 (mean ± SD)
CURVES.append(dict(cell="CHO-K1", beam="60Co", D=np.array([1, 2.5, 5, 7.5]), S=np.array([0.80, 0.55, 0.31, 0.05]),
                   sS=np.array([0.15, 0.14, 0.16, 0.01]), source="Murakami et al 2004, Table 1 (SD)"))
CURVES.append(dict(cell="CHO-K1", beam="90Sr/90Y beta", D=np.array([1, 2.5, 5, 7.5]), S=np.array([0.75, 0.56, 0.34, 0.07]),
                   sS=np.array([0.05, 0.10, 0.02, 0.01]), source="Murakami et al 2004, Table 1 (SD)"))
# Elkind & Sutton 1959 Nature 184:1293 (Chinese hamster cells), digitized from a reproduced figure; 10% assumed error
D = np.array([0.90, 1.82, 3.22, 5.00, 6.62, 8.34, 9.91, 11.44]); S = np.array([0.77, 0.60, 0.275, 0.085, 0.0193, 0.0061, 0.00182, 0.00051])
CURVES.append(dict(cell="Chinese hamster", beam="X-rays", D=D, S=S, sS=0.10*S, source="Elkind & Sutton 1959, digitized (10% assumed)"))
