import numpy as np

from cosmiclimits.neutrino.neutrino_constants import (
    FERMI_COUPLING_GEV_NEG2,
    GEV_NEG2_TO_M2,
    REFERENCE_NEUTRINO_MASS_EV,
    Z_BOSON_MASS_GEV,
    Z_BOSON_WIDTH_GEV,
)


def resonance_energy_ev(neutrino_mass_ev: float = REFERENCE_NEUTRINO_MASS_EV) -> float:
    '''
    Return the at-rest CνB Z-resonance energy E_r = M_Z^2 / (2 mν).
    Source: Ruffini, Vereshchagin, and Xue, arXiv:1503.07749, Sec. 2.3 after Eq. 2.12.
    Source: Lunardini, Sabancilar, and Yang, arXiv:1306.1808, Sec. 3 around Eq. s ≈ 2 E' m_j.
    '''
    return Z_BOSON_MASS_GEV**2 / (2.0 * neutrino_mass_ev * 1.0e-9) * 1.0e9


def cross_section_m2(
    energies_ev: np.ndarray | float,
    neutrino_mass_ev: float = REFERENCE_NEUTRINO_MASS_EV,
) -> np.ndarray:
    '''
    Compute Ruffini et al's small-momentum Breit-Wigner resonant ννbar cross section.
    Source: Ruffini, Vereshchagin, and Xue, arXiv:1503.07749, Eq. 2.12.
    Source: Lunardini, Sabancilar, and Yang, arXiv:1306.1808, Appendix A Eqns. diffcross-sigmares as the full resonant expression motivating Ruffini et al's approximation.
    Note: Uses the at-rest/small-CνB-momentum approximation selected by Ruffini, not the full thermal momentum convolution.
    '''
    energy_gev = np.asarray(energies_ev, dtype=float) * 1.0e-9
    mass_gev = neutrino_mass_ev * 1.0e-9
    xi = (Z_BOSON_WIDTH_GEV / Z_BOSON_MASS_GEV) ** 2
    numerator = (
        4.0
        * np.sqrt(2.0)
        * FERMI_COUPLING_GEV_NEG2
        * mass_gev
        * Z_BOSON_MASS_GEV**2
        * np.sqrt(xi)
        * energy_gev
    )
    denominator = (
        (Z_BOSON_MASS_GEV**2 - 2.0 * energy_gev * mass_gev) ** 2
        + 4.0 * energy_gev**2 * mass_gev**2 * xi
    )
    return numerator / denominator * GEV_NEG2_TO_M2
