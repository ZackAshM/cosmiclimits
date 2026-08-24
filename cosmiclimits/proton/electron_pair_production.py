import numpy as np

from data.crpropa_runtime import import_crpropa_data_module
from cosmiclimits.backgrounds import selected_photon_fields
from cosmiclimits.proton.photopion import _proton_lorentz_factor
from cosmiclimits.utils import TabulatedRate, load_or_build_rate_table


def loss_rate_table(energies_ev: np.ndarray) -> TabulatedRate:
    '''
    Build the proton electron-pair-production energy-loss rate table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_pairproduction.py
    '''
    energies = np.asarray(energies_ev, dtype=float)

    def build() -> np.ndarray:
        pair_production = import_crpropa_data_module("calc_pairproduction")
        gamma = _proton_lorentz_factor(energies)
        total = np.zeros_like(energies)
        for field in selected_photon_fields(include_radio=False):
            print(f"  proton pair-production loss on {field.name}", flush=True)
            total += pair_production.lossRate(gamma, field)[0]
        return total

    field_names = "_".join([f.name for f in selected_photon_fields(include_radio=False)])
    return load_or_build_rate_table(f"proton_electron_pair_production_loss_on_{field_names}", energies, build)
