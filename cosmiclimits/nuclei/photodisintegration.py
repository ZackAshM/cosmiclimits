import numpy as np

from data.crpropa_runtime import import_crpropa_data_module
from cosmiclimits.backgrounds import selected_photon_fields
from cosmiclimits.nuclei.iron import iron_lorentz_factor
from cosmiclimits.nuclei.nuclei_constants import IRON_N, IRON_Z
from cosmiclimits.utils import TabulatedRate, load_or_build_rate_table


def _iron_cross_section_table() -> tuple[np.ndarray, np.ndarray]:
    '''
    Load the Fe-56 photodisintegration total cross section.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_photodisintegration.py and https://github.com/CRPropa/CRPropa3-data/tree/master/tables/PD_Talys1.8_Khan
    '''
    photodisintegration = import_crpropa_data_module("calc_photodisintegration")
    selected = (photodisintegration.d2sum["Z"] == IRON_Z) & (photodisintegration.d2sum["N"] == IRON_N)
    if not np.any(selected):
        raise ValueError("Fe-56 photodisintegration cross section was not found in CRPropa3-data.")
    return photodisintegration.eps2, photodisintegration.xs2sum[selected][0]


def rate_table(energies_ev: np.ndarray) -> TabulatedRate:
    '''
    Build the Fe-56 photodisintegration interaction mean-free-path rate table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_photodisintegration.py and https://github.com/CRPropa/CRPropa3-data/blob/master/interactionRate.py
    '''
    energies = np.asarray(energies_ev, dtype=float)

    def build() -> np.ndarray:
        interaction_rate = import_crpropa_data_module("interactionRate")
        rest_frame_energy_j, cross_section_m2 = _iron_cross_section_table()
        gamma = iron_lorentz_factor(energies)
        total = np.zeros_like(energies)
        for field in selected_photon_fields(include_radio=True):
            print(f"  iron photodisintegration on {field.name}", flush=True)
            total += interaction_rate.calc_rate_eps(rest_frame_energy_j, cross_section_m2, gamma, field)
        return total

    field_names = "_".join([f.name for f in selected_photon_fields(include_radio=True)])
    return load_or_build_rate_table(f"iron_photodisintegration_fe56_on_{field_names}", energies, build)
