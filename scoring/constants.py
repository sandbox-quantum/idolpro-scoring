# from https://github.com/ccsb-scripps/AutoDock-Vina/blob/develop/src/lib/atom_constants.h
vina_params = {
    "vdw_radii": {
        "C": 1.9,
        "A": 1.9,
        "N": 1.8,
        "O": 1.7,
        "F": 1.5,
        "P": 2.1,
        "S": 2.0,
        "Cl": 1.8,
        "Br": 2.0,
        "I": 2.2,
    },
    "weights": {
        "w1": -0.035579,
        "w2": -0.005156,
        "w3": 0.840245,
        "w4": -0.035069,
        "w5": -0.587439,
        "w6": 0.05846,
    },
    "cutoffs": {
        "s1": 0.5,
        "o1": 0.0,
        "o2": 3.0,
        "s2": 2.0,
        "p1": 0.5,
        "p2": 1.5,
        "h1": -0.7,
    },
}
vinardo_params = {
    "vdw_radii": {"C": 2.0, "N": 1.7, "O": 1.6},
    "weights": {
        "w1": -0.045,
        "w2": 0.000,
        "w3": 0.800,
        "w4": -0.035,
        "w5": -0.600,
        "w6": 0.02,
    },
    "cutoffs": {"s1": 0.8, "p1": 0.0, "p2": 2.5, "h1": -0.6},
}

covalent_radii = {
    "H": 0.407,
    "C": 0.847,
    "N": 0.825,
    "O": 0.803,
    "F": 0.781,
    "P": 1.166,
    "S": 1.122,
    "Cl": 1.089,
    "Br": 1.254,
    "I": 1.463,
}  # includes bond allowance factor

PERIODIC_TABLE = (
    ["Dummy"]
    + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()
)
