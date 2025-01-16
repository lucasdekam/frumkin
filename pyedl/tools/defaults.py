"""
Constants not included in scipy.constants
"""

WATER_DENSITY = (
    0.99704702  # water density @25C, g/mL (https://en.wikipedia.org/wiki/Water)
)
WATER_MOLAR_MASS = (
    18.015  # water molar mass, g/mol (https://en.wikipedia.org/wiki/Water)
)
WATER_BULK_M = WATER_DENSITY / WATER_MOLAR_MASS * 1e3  # bulk water, mol/L
WATER_REL_EPS = 78.4  # water bulk permittivity @25C, referenced to eps_0
# https://en.wikipedia.org/wiki/Relative_permittivity
WATER_REL_ELEC_EPS = 1.33**2  # water optical/electronic permittivity @20C
# based on refractive index (https://en.wikipedia.org/wiki/Water)
DEFAULT_TEMPERATURE = 298  # K
