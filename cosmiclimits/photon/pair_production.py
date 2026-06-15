import numpy as np

from cosmiclimits.photon.backgrounds import selected_photon_fields
from cosmiclimits.photon.crpropa_electromagnetic import electromagnetic_rate_per_mpc
from cosmiclimits.utils import TabulatedRate, load_or_build_rate_table


def rate_table(energies_ev: np.ndarray) -> TabulatedRate:
    '''
    Build the photon pair-production absorption rate table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_electromagnetic.py
    '''
    energies = np.asarray(energies_ev, dtype=float)

    def build() -> np.ndarray:
        total = np.zeros_like(energies)
        for field in selected_photon_fields():
            print(f"  photon pair production on {field.name}", flush=True)
            total += electromagnetic_rate_per_mpc(energies, field, "pair")
        return total

    return load_or_build_rate_table("photon_pair_production", energies, build)
