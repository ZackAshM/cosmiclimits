import numpy as np

from cosmiclimits.neutrino.neutrino_constants import (
    HIGH_ENERGY_NONRESONANT_MICROBARN,
    MICROBARN_TO_M2,
    REFERENCE_NEUTRINO_MASS_EV,
)
from cosmiclimits.neutrino.z_resonance import resonance_energy_ev


def cross_section_m2(
    energies_ev: np.ndarray | float,
    neutrino_mass_ev: float = REFERENCE_NEUTRINO_MASS_EV,
) -> np.ndarray:
    '''
    Compute Ruffini et al's smooth non-resonant high-energy ννbar cross-section approximation.
    Source: Ruffini, Vereshchagin, and Xue, arXiv:1503.07749, Eq. 2.14.
    Source: Lunardini, Sabancilar, and Yang, arXiv:1306.1808, Appendix A non-resonant cross sections and Sec. 3 estimate σ_nr ≈ 8.3e-34 cm^2.
    Note: Ruffini replaces the full non-resonant channel sum by σ_he / [1 + (E/E_r)^-1].
    '''
    energies = np.asarray(energies_ev, dtype=float)
    high_energy_cross_section = HIGH_ENERGY_NONRESONANT_MICROBARN * MICROBARN_TO_M2
    return high_energy_cross_section / (1.0 + resonance_energy_ev(neutrino_mass_ev) / energies)
