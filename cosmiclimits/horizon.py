import numpy as np
from scipy.integrate import cumulative_trapezoid

from cosmiclimits.shared_constants import D_H_MPC, OMEGA_L, OMEGA_M, OMEGA_R
from cosmiclimits.utils import TabulatedRate


def e_z(redshift: np.ndarray) -> np.ndarray:
    '''
    Compute dimensionless H(z)/H0 for flat cosmology.
    Source: Eqn 3.7 of Ruffini, Vereshchagin, and Xue, "Cosmic absorption of ultra high energy particles", (https://arxiv.org/pdf/1503.07749)
    '''
    return np.sqrt(OMEGA_R * (1.0 + redshift) ** 4 + OMEGA_M * (1.0 + redshift) ** 3 + OMEGA_L)


def make_redshift_grid(z_max: float, count: int = 1000) -> np.ndarray:
    '''
    Build a redshift grid for optical-depth integration.
    '''
    return np.concatenate(([0.0], np.logspace(-8, np.log10(z_max), count)))


def cumulative_observable_distance_mpc(redshift_grid: np.ndarray) -> np.ndarray:
    '''
    Convert redshift samples to accumulated light-travel/path distance in Mpc.
    Source: Eqns 3.6 & 3.7 of Ruffini, Vereshchagin, and Xue, "Cosmic absorption of ultra high energy particles", (https://arxiv.org/pdf/1503.07749)
    '''
    integrand = D_H_MPC / ((1.0 + redshift_grid) * e_z(redshift_grid))
    return cumulative_trapezoid(integrand, redshift_grid, initial=0.0)


def cumulative_comoving_distance_mpc(redshift_grid: np.ndarray) -> np.ndarray:
    '''
    Convert redshift samples to line-of-sight comoving distance in Mpc.
    Source: Eq. 14 of Hogg, "Distance measures in cosmology", arXiv:astro-ph/9905116, https://ned.ipac.caltech.edu/level5/Hogg/Hogg4.html
    '''
    integrand = D_H_MPC / e_z(redshift_grid)
    return cumulative_trapezoid(integrand, redshift_grid, initial=0.0)


def horizon_redshift(
    observed_energy_ev: float,
    rate: TabulatedRate,
    z_max: float,
    redshift_grid: np.ndarray,
) -> float:
    '''
    Solve depth(E,z)=1 with redshift-scaled local interaction rates.
    Note: rate(E) is already a CRPropa-derived local z=0 rate table, so implementation is redshift-scaled local rate
    Source: Eqns 3.8 - 3.11 of Ruffini, Vereshchagin, and Xue, "Cosmic absorption of ultra high energy particles", (https://arxiv.org/pdf/1503.07749)
    '''
    shifted_energy = observed_energy_ev * (1.0 + redshift_grid) ** 2
    scaled_rate = (1.0 + redshift_grid) ** 3 * rate(shifted_energy)
    path_per_z_mpc = D_H_MPC / ((1.0 + redshift_grid) * e_z(redshift_grid))
    depth = cumulative_trapezoid(scaled_rate * path_per_z_mpc, redshift_grid, initial=0.0)
    if depth[-1] < 1.0:
        return z_max
    return float(np.interp(1.0, depth, redshift_grid))
