from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from data.crpropa_runtime import PACKAGE_DIR as DATA_DIR

CACHE_VERSION = 2
GENERATED_TABLE_DIR = DATA_DIR / "generated_tables"


@dataclass(frozen=True)
class TabulatedRate:
    '''
    - Holds a sampled interaction/attenuation rate table (e.g. energies_ev and rates_per_mpc)
    - Used so processes modules can return a common callable object.
    '''
    energies_ev: np.ndarray
    rates_per_mpc: np.ndarray

    def __post_init__(self) -> None:
        '''
        Validate and store positive tabulated rates.
        '''
        energies = np.asarray(self.energies_ev, dtype=float)
        rates = np.asarray(self.rates_per_mpc, dtype=float)
        if energies.ndim != 1 or rates.ndim != 1:
            raise ValueError("TabulatedRate expects one-dimensional arrays.")
        if energies.shape != rates.shape:
            raise ValueError("Energy and rate arrays must have the same shape.")
        if np.any(energies <= 0.0):
            raise ValueError("Energy samples must be positive.")
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "rates_per_mpc", np.clip(rates, 1.0e-300, None))

    def __call__(self, energies_ev: np.ndarray | float) -> np.ndarray:
        '''
        Evaluate tabulated rates by log-log interpolation.
        '''
        requested = np.asarray(energies_ev, dtype=float)
        clipped = np.clip(requested, self.energies_ev[0], self.energies_ev[-1])
        return np.exp(
            np.interp(
                np.log(clipped),
                np.log(self.energies_ev),
                np.log(self.rates_per_mpc),
            )
        )


def load_or_build_rate_table(
    cache_name: str,
    energies_ev: np.ndarray,
    builder: Callable[[], np.ndarray],
) -> TabulatedRate:
    '''
    Handles generated table data. Checks for an already existing requested table
    or generates one if not.
    '''
    GENERATED_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = GENERATED_TABLE_DIR / f"{cache_name}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if (
            int(cached["cache_version"]) == CACHE_VERSION
            and np.array_equal(cached["energies_ev"], energies_ev)
        ):
            return TabulatedRate(cached["energies_ev"], cached["rates_per_mpc"])

    rates_per_mpc = builder()
    np.savez(
        cache_path,
        cache_version=CACHE_VERSION,
        energies_ev=energies_ev,
        rates_per_mpc=rates_per_mpc,
    )
    return TabulatedRate(energies_ev, rates_per_mpc)


def static_length_mpc(rates_per_mpc: np.ndarray) -> np.ndarray:
    '''
    Convert local interaction rates to no-redshift lengths.
    '''
    return 1.0 / np.clip(rates_per_mpc, 1.0e-300, None)
