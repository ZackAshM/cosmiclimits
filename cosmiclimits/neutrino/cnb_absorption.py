import numpy as np
from scipy.integrate import cumulative_trapezoid

from cosmiclimits.horizon import e_z
from cosmiclimits.neutrino.neutrino_constants import (
    CM3_TO_M3,
    MPC_M,
    REFERENCE_NEUTRINO_MASS_EV,
    RELIC_NEUTRINO_DENSITY_CM3,
)
from cosmiclimits.neutrino.nonresonant_scattering import cross_section_m2 as nonresonant_cross_section_m2
from cosmiclimits.neutrino.z_resonance import cross_section_m2 as resonance_cross_section_m2
from cosmiclimits.shared_constants import D_H_MPC


def total_cross_section_m2(
    energies_ev: np.ndarray | float,
    neutrino_mass_ev: float = REFERENCE_NEUTRINO_MASS_EV,
) -> np.ndarray:
    '''
    Sum Ruffini et al's resonant and non-resonant ννbar cross sections.
    Source: Ruffini, Vereshchagin, and Xue, arXiv:1503.07749, Sec. 2.3 Eqns. 2.12 and 2.14.
    '''
    return resonance_cross_section_m2(energies_ev, neutrino_mass_ev) + nonresonant_cross_section_m2(
        energies_ev,
        neutrino_mass_ev,
    )


def optical_depth(
    observed_energy_ev: float,
    redshift_grid: np.ndarray,
    neutrino_mass_ev: float = REFERENCE_NEUTRINO_MASS_EV,
) -> np.ndarray:
    '''
    Compute τνν(E,z) for UHE neutrino absorption on the CνB.
    Source: Ruffini, Vereshchagin, and Xue, arXiv:1503.07749, Eq. 5.22.
    Source: Lunardini, Sabancilar, and Yang, arXiv:1306.1808, Eq. tau.
    Note: Uses Ruffini et al's n0,ν(1+z)^3 density scaling and E' = E(1+z); the 1306.1808 thermal convolution is not included.
    '''
    shifted_energy = observed_energy_ev * (1.0 + redshift_grid)
    density_m3 = RELIC_NEUTRINO_DENSITY_CM3 * CM3_TO_M3 * (1.0 + redshift_grid) ** 3
    rate_per_mpc = density_m3 * total_cross_section_m2(shifted_energy, neutrino_mass_ev) * MPC_M
    path_per_z_mpc = D_H_MPC / ((1.0 + redshift_grid) * e_z(redshift_grid))
    return cumulative_trapezoid(rate_per_mpc * path_per_z_mpc, redshift_grid, initial=0.0)


def horizon_redshift(
    observed_energy_ev: float,
    redshift_grid: np.ndarray,
    neutrino_mass_ev: float = REFERENCE_NEUTRINO_MASS_EV,
) -> float:
    '''
    Solve τνν(E,z)=1 for the neutrino horizon redshift.
    Source: Ruffini, Vereshchagin, and Xue, arXiv:1503.07749, Eq. 5.22 and Sec. 5.3.
    '''
    depth = optical_depth(observed_energy_ev, redshift_grid, neutrino_mass_ev)
    if depth[-1] < 1.0:
        return float(redshift_grid[-1])
    return float(np.interp(1.0, depth, redshift_grid))
