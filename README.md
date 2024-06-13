# Double layer modelling

This repository contains the code that was used to produce the results in the paper:

Lucas B.T. de Kam, Thomas L. Maier, Katharina Krischer,
Electrolyte effects on the alkaline hydrogen evolution reaction: A mean-field approach,
_Electrochimica Acta_,
2024,
144530,
ISSN 0013-4686,
https://doi.org/10.1016/j.electacta.2024.144530.


## Abstract

This paper introduces the combination of an advanced double-layer model with electrochemical kinetics to explain electrolyte effects on the alkaline hydrogen evolution reaction. It is known from experimental studies that the alkaline hydrogen evolution current shows a strong dependence on the concentration and identity of cations in the electrolyte, but is independent of pH. To explain these effects, we formulate the faradaic current in terms of the electric potential in the double layer, which is calculated using a mean-field model that takes into account the cation and anion sizes as well as the electric dipole moment of water molecules. We propose that the Volmer step consists of two activated processes: a water reduction sub-step, and a sub-step in which OH− is transferred away from the reaction plane through the double layer. Either of these sub-steps may limit the rate. The proposed models for these sub-steps qualitatively explain experimental observations, including cation effects, pH-independence, and the trend reversal between gold and platinum electrodes. We also assess the quantitative accuracy of the water-reduction-limited current model; we suggest that the predicted functional relationship is valid as long as the hydrogen bonding structure of water near the electrode is sufficiently maintained.

## The code

The model code is in the folder `edl`. Scripts that plot the results from the paper are named `figure_*.py`. For the scatter data plots, data was extracted from figures of other publications (as cited in the paper); the extracted values are in the folder `data`. This data is processed further by the `prepare_scatter_data_*.ipynb` notebooks. The resulting dataframes stored in `data/*_df.csv` are used to generate the scatter plots.

