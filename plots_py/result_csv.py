import numpy as np

datasets = {"3-Feb":"Biomass_23-02-03.xlsx",
            "8-Feb":"Biomass_23-02-08.xlsx",
            "16-Feb": "Biomass_23-02-16.xlsx",
            "23-Feb": "Biomass_23-02-23.xlsx",
            "8-Mar": "Biomass_23-03-08.xlsx",
            "15-Mar": "Biomass_23-03-15.xlsx",
            "22-Mar": "Biomass_23-03-22.xlsx",
            "3-Apr": "Biomass_23-04-03.xlsx",}

x_days = {"25-Jan":   0,
          "3-Feb":   9,
          "8-Feb":  14,
          "16-Feb":  22,
          "23-Feb":  29,
          "8-Mar":  43,
          "15-Mar":  50,
          "22-Mar":  57,
          "3-Apr":  69,}

color_maps = {
    "ours": "b",
    "dvi": "g",
    "bionet": "r",
    "dgcnn": "c",
    "pnetpp": "m",
    "pnet": "y",
}

key2names = {
    "ours": "Ours",
    "dvi": "3DVI",
    "bionet": "PC-BioNet",
    "dgcnn": "DGCNN",
    "pnetpp": "PointNet++",
    "pnet": "PointNet",
}

ours = {"3-Feb":   0.2757,
       "8-Feb":   0.1897,
       "16-Feb":  0.2297,
       "23-Feb":  0.1986,
       "8-Mar":   0.1910,
       "15-Mar":  0.1749,
       "22-Mar":  0.2051,
       "3-Apr":   0.3765,}

bionet={"3-Feb":   0.2567,
        "8-Feb":   0.2128,
        "16-Feb":  0.1981,
        "23-Feb":  0.1926,
        "8-Mar":   0.2010,
        "15-Mar":  0.1849,
        "22-Mar":  0.2351,
        "3-Apr":   0.3665,
}
dgcnn ={"3-Feb":   0.3557,
        "8-Feb":   0.2656,
        "16-Feb":  0.2413,
        "23-Feb":  0.2286,
        "8-Mar":   0.2340,
        "15-Mar":  0.2169,
        "22-Mar":  0.2561,
        "3-Apr":   0.4165,
}

