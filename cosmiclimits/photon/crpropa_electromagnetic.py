import math

import numpy as np

from data.crpropa_runtime import import_crpropa_data_module
from cosmiclimits.photon.photon_constants import (
    DEFAULT_MAX_LOG10_S_KIN_EV2,
    EM_ROMBERG_EXPONENT,
    MIN_LOG10_S_KIN_EV2,
)


def electromagnetic_rate_per_mpc(
    energies_ev: np.ndarray,
    field,
    process_name: str,
) -> np.ndarray:
    '''
    Calculate a photon electromagnetic interaction rate for one photon field.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_electromagnetic.py and https://github.com/CRPropa/CRPropa3-data/blob/master/interactionRate.py
    Extends the upper s_kin grid when needed to cover the plotted energy range.
    '''
    electromagnetic = import_crpropa_data_module("calc_electromagnetic")
    interaction_rate = import_crpropa_data_module("interactionRate")
    units = import_crpropa_data_module("units")

    sigma_by_process = {
        "pair": electromagnetic.sigmaPP,
        "double_pair": electromagnetic.sigmaDPP,
    }
    sigma = sigma_by_process[process_name]

    max_background_energy_ev = field.getEmax() / units.eV
    max_energy_ev = float(np.max(energies_ev))
    max_log10_s_kin_ev2 = max(
        DEFAULT_MAX_LOG10_S_KIN_EV2,
        math.ceil(math.log10(4.0 * max_energy_ev * max_background_energy_ev)) + 1.0,
    )
    s_kin = np.logspace(
        MIN_LOG10_S_KIN_EV2,
        max_log10_s_kin_ev2,
        2**EM_ROMBERG_EXPONENT + 1,
    ) * units.eV**2
    cross_section_m2 = electromagnetic.getTabulatedXS(sigma, s_kin)
    energies_j = energies_ev * units.eV
    return interaction_rate.calc_rate_s(s_kin, cross_section_m2, energies_j, field)
