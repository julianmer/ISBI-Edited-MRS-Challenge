####################################################################################################
#                                        simulationDefs.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Simulate a corpus of data sets for a metabolite basis set with a specific range of      #
#          concentration values and noise.                                                         #
#                                                                                                  #
####################################################################################################


#**************************#
#   concentration ranges   #
#**************************#
stdConcs = {
    # due to different naming conventions some metabolites
    # might be defined multiple times
    'Ace': {'name': 'Ace', 'low_limit': 0, 'up_limit': 0},  # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0.1, 'up_limit': 1.5},  # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0, 'up_limit': 0},  # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 2.0},  # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 4.5, 'up_limit': 10.5},  # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 1.0, 'up_limit': 2.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 1.0, 'up_limit': 2.0},  # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 3.0, 'up_limit': 6.0},  # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 6.0, 'up_limit': 12.5},  # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0, 'up_limit': 0},  # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.5, 'up_limit': 2.0},  # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 1.5, 'up_limit': 3.0},  # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 0, 'up_limit': 0},  #
    'Lac': {'name': 'Lac', 'low_limit': 0.2, 'up_limit': 1.0},  # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 1.0, 'up_limit': 2.0},  # Macromolecules
    'mI': {'name': 'mI', 'low_limit': 4.0, 'up_limit': 9.0},  # Myo-inositol
    'NAA': {'name': 'NAA', 'low_limit': 7.5, 'up_limit': 17.0},  # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.5, 'up_limit': 2.5},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0, 'up_limit': 0},  #
    'PC': {'name': 'PC', 'low_limit': 0.5, 'up_limit': 2.0},  #
    'PCr': {'name': 'PCr', 'low_limit': 3.0, 'up_limit': 5.5},  # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 1.0, 'up_limit': 2.0},  # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 2.0, 'up_limit': 6.0},  # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0.0, 'up_limit': 0.0},  # Scyllo-inositol

    'Ch': {'name': 'Ch', 'low_limit': 0.0, 'up_limit': 0.0},  # Choline
    'Cho': {'name': 'Cho', 'low_limit': 0.0, 'up_limit': 0.0},  # Choline
    'Eth': {'name': 'Eth', 'low_limit': 0.0, 'up_limit': 0.0},  # Ethanolamine
    'Hom': {'name': 'Hom', 'low_limit': 0.0, 'up_limit': 0.0},  # Homocarnosine

    'Glx': {'name': 'Glx', 'low_limit': 9.0, 'up_limit': 18.5},
    'tCho': {'name': 'tCho', 'low_limit': 1.0, 'up_limit': 4.0},
    'tCr': {'name': 'tCr', 'low_limit': 7.5, 'up_limit': 16.0},
    'tNAA': {'name': 'tNAA', 'low_limit': 8.0, 'up_limit': 19.5},

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 0},  # Scyllo?
    'PCh': {'name': 'PCh', 'low_limit': 0, 'up_limit': 0},  #
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 1., 'up_limit': 2.0},  # Macromolecules

    'Ch': {'name': 'Ch', 'low_limit': 0., 'up_limit': 0},
    'Cit': {'name': 'Cit', 'low_limit': 0., 'up_limit': 0},
    'EtOH': {'name': 'EtOH', 'low_limit': 0., 'up_limit': 0},
    'H2O': {'name': 'H2O', 'low_limit': 400, 'up_limit': 600},
    'Phenyl': {'name': 'Phenyl', 'low_limit': 0., 'up_limit': 0},
    'Ref0ppm': {'name': 'Ref0ppm', 'low_limit': 0., 'up_limit': 0},
    'Ser': {'name': 'Ser', 'low_limit': 0., 'up_limit': 0},
    'Tyros': {'name': 'Tyros', 'low_limit': 0., 'up_limit': 0},
    'bHB': {'name': 'bHB', 'low_limit': 0., 'up_limit': 0},
    'bHG': {'name': 'bHG', 'low_limit': 0., 'up_limit': 0},
}

fitConcs = {
    'Ace': {'name': 'Ace', 'low_limit': 0., 'up_limit': 3.},  # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0., 'up_limit': 8.},  # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0., 'up_limit': 7.},  # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 8.0},  # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 0., 'up_limit': 9.},  # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0., 'up_limit': 9.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 0., 'up_limit': 3.0},  # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 1.0, 'up_limit': 14.0},  # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 4.0, 'up_limit': 15.0},  # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0., 'up_limit': 9.0},  # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0., 'up_limit': 3.0},  # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0., 'up_limit': 4.0},  # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 1.0, 'up_limit': 12.0},  #
    'Lac': {'name': 'Lac', 'low_limit': 0., 'up_limit': 38.0},  # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 0., 'up_limit': 14.0},  # Macromolecules
    'NAA': {'name': 'NAA', 'low_limit': 0., 'up_limit': 18.0},  # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0., 'up_limit': 4.0},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0., 'up_limit': 3.0},  #
    'PCr': {'name': 'PCr', 'low_limit': 0., 'up_limit': 7.0},  # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0., 'up_limit': 4.0},  # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 0., 'up_limit': 4.0},  # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0., 'up_limit': 2.0},  # Scyllo-inositol

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 2.0},  # Scyllo?
    'PCh': {'name': 'PCh', 'low_limit': 0., 'up_limit': 3.0},  #
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 0., 'up_limit': 14.0},  # Macromolecules

    'Ch': {'name': 'Ch', 'low_limit': 0., 'up_limit': 0},
    'Cit': {'name': 'Cit', 'low_limit': 0., 'up_limit': 0},
    'EtOH': {'name': 'EtOH', 'low_limit': 0., 'up_limit': 0},
    'H2O': {'name': 'H2O', 'low_limit': 200, 'up_limit': 1000},
    'Phenyl': {'name': 'Phenyl', 'low_limit': 0., 'up_limit': 0},
    'Ref0ppm': {'name': 'Ref0ppm', 'low_limit': 0., 'up_limit': 0},
    'Ser': {'name': 'Ser', 'low_limit': 0., 'up_limit': 0},
    'Tyros': {'name': 'Tyros', 'low_limit': 0., 'up_limit': 0},
    'bHB': {'name': 'bHB', 'low_limit': 0., 'up_limit': 0},
    'bHG': {'name': 'bHG', 'low_limit': 0., 'up_limit': 0},
}


allConcs = {
    'Ace': {'name': 'Ace', 'low_limit': 0., 'up_limit': 25.},  # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0., 'up_limit': 25.},  # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0., 'up_limit': 25.},  # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 0, 'up_limit': 25.0},  # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 0., 'up_limit': 25.},  # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0., 'up_limit': 25.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 0., 'up_limit': 25.0},  # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 0, 'up_limit': 25.0},  # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 0, 'up_limit': 25.0},  # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0., 'up_limit': 25.0},  # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0., 'up_limit': 25.0},  # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0., 'up_limit': 25.0},  # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 0, 'up_limit': 25.0},  #
    'Lac': {'name': 'Lac', 'low_limit': 0., 'up_limit': 25.0},  # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 0., 'up_limit': 25.0},  # Macromolecules
    'NAA': {'name': 'NAA', 'low_limit': 0., 'up_limit': 25.0},  # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0., 'up_limit': 25.0},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0., 'up_limit': 25.0},  #
    'PCr': {'name': 'PCr', 'low_limit': 0., 'up_limit': 25.0},  # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0., 'up_limit': 25.0},  # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 0., 'up_limit': 25.0},  # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0., 'up_limit': 25.0},  # Scyllo-inositol

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 25.0},  # Scyllo?
    'PCh': {'name': 'PCh', 'low_limit': 0., 'up_limit': 25.0},  #
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 0., 'up_limit': 25.0},  # Macromolecules

    'Ch': {'name': 'Ch', 'low_limit': 0., 'up_limit': 25},
    'Cit': {'name': 'Cit', 'low_limit': 0., 'up_limit': 25},
    'EtOH': {'name': 'EtOH', 'low_limit': 0., 'up_limit': 25},
    'H2O': {'name': 'H2O', 'low_limit': 0, 'up_limit': 2500},
    'Phenyl': {'name': 'Phenyl', 'low_limit': 0., 'up_limit': 25},
    'Ref0ppm': {'name': 'Ref0ppm', 'low_limit': 0., 'up_limit': 25},
    'Ser': {'name': 'Ser', 'low_limit': 0., 'up_limit': 25},
    'Tyros': {'name': 'Tyros', 'low_limit': 0., 'up_limit': 25},
    'bHB': {'name': 'bHB', 'low_limit': 0., 'up_limit': 25},
    'bHG': {'name': 'bHG', 'low_limit': 0., 'up_limit': 25},

}


customConcs = {
    'Ace': {'name': 'Ace', 'low_limit': 0, 'up_limit': 0.1},  # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0.1, 'up_limit': 1.5},  # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0, 'up_limit': 0.1},  # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 2.0},  # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 4.5, 'up_limit': 10.5},  # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 1.0, 'up_limit': 3.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 1.0, 'up_limit': 2.0},  # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 3.0, 'up_limit': 6.0},  # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 6.0, 'up_limit': 12.5},  # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0, 'up_limit': 0.1},  # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.5, 'up_limit': 2.0},  # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 1.5, 'up_limit': 3.0},  # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 0, 'up_limit': 0.1},  #
    'Lac': {'name': 'Lac', 'low_limit': 0.2, 'up_limit': 1.0},  # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 1.0, 'up_limit': 2.0},  # Macromolecules
    'mI': {'name': 'mI', 'low_limit': 4.0, 'up_limit': 9.0},  # Myo-inositol
    'NAA': {'name': 'NAA', 'low_limit': 7.5, 'up_limit': 17.0},  # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.5, 'up_limit': 2.5},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0, 'up_limit': 0.1},  #
    'PC': {'name': 'PC', 'low_limit': 0.5, 'up_limit': 2.0},  #
    'PCr': {'name': 'PCr', 'low_limit': 3.0, 'up_limit': 5.5},  # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 1.0, 'up_limit': 2.0},  # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 2.0, 'up_limit': 6.0},  # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0.0, 'up_limit': 0.1},  # Scyllo-inositol

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 0.1},  # Scyllo?
    'PCh': {'name': 'PCh', 'low_limit': 0, 'up_limit': 0.1},  #
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 1., 'up_limit': 2.0},  # Macromolecules

    'Ch': {'name': 'Ch', 'low_limit': 0., 'up_limit': 0.1},
    'Cit': {'name': 'Cit', 'low_limit': 0., 'up_limit': 0.1},
    'EtOH': {'name': 'EtOH', 'low_limit': 0., 'up_limit': 0.1},
    'H2O': {'name': 'H2O', 'low_limit': 400, 'up_limit': 600},
    'Phenyl': {'name': 'Phenyl', 'low_limit': 0., 'up_limit': 0.1},
    'Ref0ppm': {'name': 'Ref0ppm', 'low_limit': 0., 'up_limit': 0.1},
    'Ser': {'name': 'Ser', 'low_limit': 0., 'up_limit': 0.1},
    'Tyros': {'name': 'Tyros', 'low_limit': 0., 'up_limit': 0.1},
    'bHB': {'name': 'bHB', 'low_limit': 0., 'up_limit': 0.1},
    'bHG': {'name': 'bHG', 'low_limit': 0., 'up_limit': 0.1},
}


#********************#
#   initialization   #
#********************#
cleanParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (2, 2)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    'ownBaseline': None,
    'noiseCov': [0, 0],  # [low, high]
    'bandwidth': 2000,
    'dwelltime': 1 / 2000,
    'centralFrequency': 127.731,
    'points': 2048,
}

fitParams = {
    'dist': 'unif',
    'broadening': [(1, 0), (35, 25)],  # [(low, low), (high, high)]
    'coilamps': [0.5, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 1.5],  # [low, high]
    'phi1': [-1e-4, 1e-4],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-3, -4, -5, -3, -8, -2], [1, 2, 3, 5, 1, 5]],
    'ownBaseline': None,
    'noiseCov': [5, 50],  # [low, high]
    'bandwidth': 2000,
    'dwelltime': 1 / 2000,
    'centralFrequency': 127.731,
    'points': 2048,
}

badParams = {
    'dist': 'unif',
    'broadening': [(0, 0), (50, 50)],  # [(low, low), (high, high)]
    'coilamps': [0.5, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-1.5, 1.5],  # [low, high]
    'phi1': [-1e-4, 1e-4],  # [low, high]
    'shifting': [-20, 20],  # [low, high]
    'baseline': [[-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10]],
    'ownBaseline': None,
    'noiseCov': [1, 100],  # [low, high]
    'bandwidth': 2000,
    'dwelltime': 1 / 2000,
    'centralFrequency': 127.731,
    'points': 2048,
}

customParams = {
    'dist': 'unif',
    'broadening': [(6, 6), (12, 12)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    'ownBaseline': None,
    'noiseCov': [0, 0],  # [low, high]
    'bandwidth': 2000,
    'dwelltime': 1 / 2000,
    'centralFrequency': 127.731,
    'points': 2048,
}