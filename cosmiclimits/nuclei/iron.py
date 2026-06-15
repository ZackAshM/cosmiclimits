from pathlib import Path

import numpy as np

from data.crpropa_runtime import CRPROPA_DATA_DIR, import_crpropa_data_module
from cosmiclimits.nuclei.nuclei_constants import IRON_A, IRON_N, IRON_Z


def iron_rest_energy_ev() -> float:
    '''
    Return the Fe-56 nuclear rest energy from CRPropa3-data's NIST mass table.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_mass.py and https://github.com/CRPropa/CRPropa3-data/blob/master/tables/mass_NIST.txt
    '''
    units = import_crpropa_data_module("units")
    table_path = Path(CRPROPA_DATA_DIR) / "tables" / "mass_NIST.txt"
    atomic_number = None
    mass_number = None
    for line in table_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Atomic Number"):
            atomic_number = int(line.split("=")[1])
        elif line.startswith("Mass Number"):
            mass_number = int(line.split("=")[1])
        elif (
            line.startswith("Relative Atomic Mass")
            and atomic_number == IRON_Z
            and mass_number == IRON_A
        ):
            mass_text = line.split("=")[1].strip().split("(")[0]
            atomic_mass_kg = float(mass_text) * units.amu
            nuclear_mass_kg = atomic_mass_kg - IRON_Z * units.mass_electron
            return nuclear_mass_kg * units.c_squared / units.eV
    raise ValueError(f"Fe-{IRON_A} mass was not found in {table_path}.")


def iron_lorentz_factor(energies_ev: np.ndarray) -> np.ndarray:
    '''
    Convert Fe-56 total energy to Lorentz factor.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_mass.py
    '''
    return energies_ev / iron_rest_energy_ev()

