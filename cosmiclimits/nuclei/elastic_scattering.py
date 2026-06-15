import numpy as np

from data.crpropa_runtime import CRPROPA_DATA_DIR, import_crpropa_data_module
from cosmiclimits.nuclei.backgrounds import selected_photon_fields
from cosmiclimits.nuclei.iron import iron_lorentz_factor
from cosmiclimits.nuclei.nuclei_constants import IRON_A, IRON_N, IRON_Z
from cosmiclimits.utils import TabulatedRate, load_or_build_rate_table


def _iron_cross_section_table() -> tuple[np.ndarray, np.ndarray]:
    '''
    Load the Fe-56 elastic-scattering cross section.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_elasticscattering.py and https://github.com/CRPropa/CRPropa3-data/tree/master/tables/PD_Talys1.8_Khan
    '''
    interaction_rate = import_crpropa_data_module("interactionRate")
    units = import_crpropa_data_module("units")
    table_dir = CRPROPA_DATA_DIR / "tables" / "PD_Talys1.8_Khan"
    rest_frame_energy_j = np.genfromtxt(table_dir / "eps_elastic.txt") * units.eV * 1.0e6
    table = np.genfromtxt(
        table_dir / "xs_elastic.txt",
        dtype=[("Z", int), ("N", int), ("xs", f"{len(rest_frame_energy_j)}f8")],
    )
    table = table[(table["Z"] + table["N"]) >= 12]
    scaling = (table["Z"] * table["N"] / (table["Z"] + table["N"]))[:, np.newaxis]
    table["xs"] /= scaling
    selected = (table["Z"] == IRON_Z) & (table["N"] == IRON_N)
    if not np.any(selected):
        raise ValueError("Fe-56 elastic-scattering cross section was not found in CRPropa3-data.")
    rest_frame_energy_j = interaction_rate.romb_pad_logspaced(rest_frame_energy_j, 513)
    cross_section_m2 = interaction_rate.romb_pad_zero(table["xs"][selected][0], 513) * 1.0e-31
    return rest_frame_energy_j, cross_section_m2 * (IRON_Z * IRON_N / IRON_A)


def rate_table(energies_ev: np.ndarray) -> TabulatedRate:
    '''
    Build the Fe-56 elastic-scattering interaction rate table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_elasticscattering.py and https://github.com/CRPropa/CRPropa3-data/blob/master/interactionRate.py
    '''
    energies = np.asarray(energies_ev, dtype=float)

    def build() -> np.ndarray:
        interaction_rate = import_crpropa_data_module("interactionRate")
        rest_frame_energy_j, cross_section_m2 = _iron_cross_section_table()
        gamma = iron_lorentz_factor(energies)
        total = np.zeros_like(energies)
        for field in selected_photon_fields(include_radio=True):
            print(f"  iron elastic scattering on {field.name}", flush=True)
            total += interaction_rate.calc_rate_eps(rest_frame_energy_j, cross_section_m2, gamma, field)
        return total

    return load_or_build_rate_table("iron_elastic_scattering_fe56", energies, build)
