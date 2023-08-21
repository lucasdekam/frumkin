"""
Physical constants for use in double-layer models
"""

E_0 = 1.602e-19  # elementary charge
N_A = 6.022e23  # Avogadro's number
K_B = 1.38e-23  # Boltzmann's constant
T = 298  # temperature, K
Z = 1  # ion valency
BETA = 1 / K_B / T  # inverse thermal energy
EPS_0 = 8.8541878128e-12  # vacuum permittivity, F/m
EPS_R_WATER = 78.5  # relative permittivity of bulk water
C_WATER_BULK = 55.5  # water bulk concentration, molar
N_SITES_SILICA = 5e18  # surface site density on silica, /m^2
K_SILICA_A = 10 ** (-6)  # equilibrium constant silica high pH, molar
K_SILICA_B = 10 ** (2)  # equilibrium constant silica low pH, molar
K_WATER = 10 ** (-14)  # equilibrium constant water dissociation, molar^2
PKW = 14  # -log water constant
N_WATER = 1.33  # refractive index of water
D_ADSORBATE_LAYER = 0.1e-10  # adsorbate layer thickness, m
P_0 = 101325  # atmospheric pressure, Pa

# Kinetics
AU_PZC_SHE_V = 0.2  # PZC for Au in V vs. SHE in Thomas' work
PT_PZC_SHE_V = 1  # PZC for Pt(111) in V vs. SHE (approximately)
