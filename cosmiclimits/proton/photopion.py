from pathlib import Path

import numpy as np

from data.crpropa_runtime import CRPROPA_DATA_DIR, import_crpropa_data_module
from cosmiclimits.proton.backgrounds import selected_photon_fields
from cosmiclimits.proton.proton_constants import PHOTOPION_REGRID_EXPONENT
from cosmiclimits.utils import TabulatedRate, load_or_build_rate_table


def _proton_lorentz_factor(energies_ev: np.ndarray) -> np.ndarray:
    '''
    Convert proton total energy to Lorentz factor.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/units.py
    '''
    units = import_crpropa_data_module("units")
    proton_rest_energy_ev = units.mass_proton * units.c_squared / units.eV
    return energies_ev / proton_rest_energy_ev


def _load_proton_cross_section_table() -> tuple[np.ndarray, np.ndarray]:
    '''
    Load and regrid the proton photopion cross-section table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/tables/PPP/xs_proton.txt
    Uses the full shipped table rather than CRPropa3-data's default 2049-row truncation.
    '''
    units = import_crpropa_data_module("units")
    table_path = Path(CRPROPA_DATA_DIR) / "tables" / "PPP" / "xs_proton.txt"
    data = np.genfromtxt(table_path, comments="#")
    rest_frame_energy_ev = data[:, 0] * 1.0e9
    cross_section_m2 = data[:, 1] * 1.0e-34

    regridded_energy_ev = np.logspace(
        np.log10(rest_frame_energy_ev[0]),
        np.log10(rest_frame_energy_ev[-1]),
        2**PHOTOPION_REGRID_EXPONENT + 1,
    )
    regridded_cross_section_m2 = np.interp(
        np.log(regridded_energy_ev),
        np.log(rest_frame_energy_ev),
        cross_section_m2,
    )
    return regridded_energy_ev * units.eV, regridded_cross_section_m2


def rate_table(energies_ev: np.ndarray) -> TabulatedRate:
    '''
    Build the proton photopion interaction mean-free-path rate table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_photopionproduction.py and https://github.com/CRPropa/CRPropa3-data/blob/master/interactionRate.py
    '''
    energies = np.asarray(energies_ev, dtype=float)

    def build() -> np.ndarray:
        interaction_rate = import_crpropa_data_module("interactionRate")
        rest_frame_energy_j, cross_section_m2 = _load_proton_cross_section_table()
        gamma = _proton_lorentz_factor(energies)
        total = np.zeros_like(energies)
        for field in selected_photon_fields(include_radio=True):
            print(f"  proton photopion on {field.name}", flush=True)
            total += interaction_rate.calc_rate_eps(rest_frame_energy_j, cross_section_m2, gamma, field)
        return total

    return load_or_build_rate_table("proton_photopion_full_table", energies, build)
