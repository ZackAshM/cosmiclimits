import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data.crpropa_runtime import ensure_crpropa_data_path
from cosmiclimits.horizon import cumulative_observable_distance_mpc, horizon_redshift, make_redshift_grid
from cosmiclimits.nuclei.elastic_scattering import rate_table as iron_elastic_rate_table
from cosmiclimits.nuclei.photodisintegration import rate_table as iron_photodisintegration_rate_table
from cosmiclimits.photon.double_pair_production import rate_table as photon_double_pair_rate_table
from cosmiclimits.photon.pair_production import rate_table as photon_pair_rate_table
from cosmiclimits.proton.electron_pair_production import loss_rate_table as proton_pair_loss_rate_table
from cosmiclimits.proton.photopion import rate_table as proton_photopion_rate_table
from cosmiclimits.utils import TabulatedRate, static_length_mpc

LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 2.0}
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12
TITLE_SIZE = 18
LEGEND_SIZE = 12.5
IRON_COLOR = "darkorange"
IRON_LINESTYLE = "-."


def combine_rates(*tables: TabulatedRate) -> TabulatedRate:
    '''
    Sum process rates sampled on the same energy grid.
    '''
    energies = tables[0].energies_ev
    total = np.zeros_like(energies)
    for table in tables:
        if not np.array_equal(table.energies_ev, energies):
            raise ValueError("Cannot combine tables sampled on different energy grids.")
        total += table.rates_per_mpc
    return TabulatedRate(energies, total)


def compute_curves() -> dict[str, np.ndarray]:
    print("checking CRPropa3-data runtime", flush=True)
    ensure_crpropa_data_path()

    photon_energy_grid = np.logspace(10.0, 25.0, 190)
    proton_energy_grid = np.logspace(17.0, 25.0, 150)
    iron_energy_grid = np.logspace(17.0, 25.0, 150)

    print("building photon CRPropa rate tables", flush=True)
    photon_pair = photon_pair_rate_table(photon_energy_grid)
    photon_double_pair = photon_double_pair_rate_table(photon_energy_grid)
    photon_total = combine_rates(photon_pair, photon_double_pair)

    print("building proton CRPropa rate tables", flush=True)
    proton_photopion = proton_photopion_rate_table(proton_energy_grid)
    proton_pair_loss = proton_pair_loss_rate_table(proton_energy_grid)
    proton_total = combine_rates(proton_photopion, proton_pair_loss)

    print("building iron CRPropa rate tables", flush=True)
    iron_photodisintegration = iron_photodisintegration_rate_table(iron_energy_grid)
    iron_elastic = iron_elastic_rate_table(iron_energy_grid)
    iron_total = combine_rates(iron_photodisintegration, iron_elastic)

    z_grid = make_redshift_grid(40.0, count=1000)
    distance_grid = cumulative_observable_distance_mpc(z_grid)

    print("solving photon total horizon", flush=True)
    photon_z = np.array(
        [horizon_redshift(energy, photon_total, 40.0, z_grid) for energy in photon_energy_grid]
    )
    print("solving proton horizon", flush=True)
    proton_z = np.array(
        [horizon_redshift(energy, proton_total, 40.0, z_grid) for energy in proton_energy_grid]
    )
    print("solving iron horizon", flush=True)
    iron_z = np.array(
        [horizon_redshift(energy, iron_total, 40.0, z_grid) for energy in iron_energy_grid]
    )
    return {
        "photon_energy_ev": photon_energy_grid,
        "photon_z": photon_z,
        "photon_distance_mpc": np.interp(photon_z, z_grid, distance_grid),
        "photon_static_distance_mpc": static_length_mpc(photon_total.rates_per_mpc),
        "photon_pair_rate_mpc": photon_pair.rates_per_mpc,
        "photon_double_pair_rate_mpc": photon_double_pair.rates_per_mpc,
        "photon_total_rate_mpc": photon_total.rates_per_mpc,
        "proton_energy_ev": proton_energy_grid,
        "proton_z": proton_z,
        "proton_distance_mpc": np.interp(proton_z, z_grid, distance_grid),
        "proton_static_distance_mpc": static_length_mpc(proton_total.rates_per_mpc),
        "proton_photopion_rate_mpc": proton_photopion.rates_per_mpc,
        "proton_pair_loss_rate_mpc": proton_pair_loss.rates_per_mpc,
        "proton_total_rate_mpc": proton_total.rates_per_mpc,
        "iron_energy_ev": iron_energy_grid,
        "iron_z": iron_z,
        "iron_distance_mpc": np.interp(iron_z, z_grid, distance_grid),
        "iron_static_distance_mpc": static_length_mpc(iron_total.rates_per_mpc),
        "iron_photodisintegration_rate_mpc": iron_photodisintegration.rates_per_mpc,
        "iron_elastic_rate_mpc": iron_elastic.rates_per_mpc,
        "iron_total_rate_mpc": iron_total.rates_per_mpc,
    }


def add_record_lines(ax: plt.Axes, label_x: float) -> None:
    records = [
        (3.20e20, "highest cosmic-ray observation\nFly's Eye, 320 EeV", "firebrick"),
        (3.70e15, "highest photon observation\nLHAASO, 3.7 PeV", "navy"),
    ]
    for energy_ev, label, color in records:
        '''
        Convert record energies to plotted log10(E/eV) y-coordinates.
        Source: Local plotting implementation.
        '''
        y = np.log10(energy_ev)
        ax.axhline(y, color=color, linestyle=(0, (7, 4)), linewidth=1.7, alpha=0.82)
        ax.text(
            label_x,
            y + 0.08,
            label,
            color=color,
            fontsize=10,
            ha="center",
            va="bottom",
            bbox=LABEL_BOX,
        )


def plot_mpc(curves: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 7.4), constrained_layout=True)
    ax.set_xlim(-3.0, 4.6)
    ax.set_ylim(7.0, 25.0)
    ax.set_xticks(np.arange(-3.0, 5.0, 1.0))

    photon_y = np.log10(curves["photon_energy_ev"])
    proton_y = np.log10(curves["proton_energy_ev"])
    iron_y = np.log10(curves["iron_energy_ev"])
    photon_x = np.log10(np.clip(curves["photon_distance_mpc"], 1.0e-3, 10.0**4.6))
    proton_x = np.log10(np.clip(curves["proton_distance_mpc"], 1.0e-3, 10.0**4.6))
    iron_x = np.log10(np.clip(curves["iron_distance_mpc"], 1.0e-3, 10.0**4.6))

    photon_static_x = np.log10(
        np.clip(curves["photon_static_distance_mpc"], 1.0e-3, 10.0**4.6)
    )
    proton_static_x = np.log10(np.clip(curves["proton_static_distance_mpc"], 1.0e-3, 10.0**4.6))
    iron_static_x = np.log10(np.clip(curves["iron_static_distance_mpc"], 1.0e-3, 10.0**4.6))

    ax.fill_betweenx(
        photon_y,
        photon_x,
        4.6,
        facecolor="blue",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.02,
    )
    ax.fill_betweenx(
        photon_y,
        photon_x,
        4.6,
        facecolor="none",
        hatch="/",
        edgecolor="royalblue",
        linewidth=0.0,
        alpha=0.60,
    )
    ax.fill_betweenx(
        proton_y,
        proton_x,
        4.6,
        facecolor="red",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.02,
    )
    ax.fill_betweenx(
        proton_y,
        proton_x,
        4.6,
        facecolor="none",
        hatch="\\",
        edgecolor="red",
        linewidth=0.0,
        alpha=0.42,
    )

    ax.plot(photon_static_x, photon_y, color="blue", linewidth=2.0, linestyle=(0, (5, 3)), alpha=0.78)
    ax.plot(
        proton_static_x,
        proton_y,
        color="red",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        alpha=0.68,
    )
    ax.plot(
        iron_static_x,
        iron_y,
        color=IRON_COLOR,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        alpha=0.72,
    )
    ax.plot(
        photon_x,
        photon_y,
        color="blue",
        linewidth=2.4,
        label=r"$\gamma\gamma \rightarrow e^+e^-,\;\gamma\gamma \rightarrow 2(e^+e^-)$",
    )
    ax.plot(
        proton_x,
        proton_y,
        color="red",
        linewidth=2.4,
        label=r"$p\gamma \rightarrow \Delta^+ \rightarrow p\pi^0 / n\pi^+$",
    )
    ax.plot(
        iron_x,
        iron_y,
        color=IRON_COLOR,
        linewidth=2.4,
        linestyle=IRON_LINESTYLE,
        label=r"$^{56}\mathrm{Fe}+\gamma \rightarrow \mathrm{fragments}$",
    )

    add_record_lines(ax, label_x=3.0)
    ax.text(3.2, 23.0, "protons", color="red", fontsize=15, ha="center", bbox=LABEL_BOX)
    ax.text(0.5, 17.0, "photons", color="blue", fontsize=15, ha="center", bbox=LABEL_BOX)

    ax.set_xlabel(r"$\log_{10}$(Observable distance / Mpc)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\log_{10}$(particle energy / eV)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("Particle Horizon Limits", fontsize=TITLE_SIZE)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=TICK_LABEL_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=2.0, linestyle=(0, (5, 3))))
    labels.append("No Redshift Evolution")
    ax.legend(handles, labels, loc="upper left", fontsize=LEGEND_SIZE, framealpha=0.82, facecolor="white", edgecolor="none")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_redshift(curves: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 7.4), constrained_layout=True)
    ax.set_xlim(-8.0, 4.0)
    ax.set_ylim(7.0, 25.0)
    ax.set_xticks(np.arange(-8.0, 5.0, 2.0))

    photon_y = np.log10(curves["photon_energy_ev"])
    proton_y = np.log10(curves["proton_energy_ev"])
    iron_y = np.log10(curves["iron_energy_ev"])
    photon_x = np.log10(np.clip(curves["photon_z"], 1.0e-7, 10.0**4.0))
    proton_x = np.log10(np.clip(curves["proton_z"], 1.0e-7, 10.0**4.0))
    iron_x = np.log10(np.clip(curves["iron_z"], 1.0e-7, 10.0**4.0))

    ax.fill_betweenx(
        photon_y,
        photon_x,
        4.0,
        facecolor="blue",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.02,
    )
    ax.fill_betweenx(
        photon_y,
        photon_x,
        4.0,
        facecolor="none",
        hatch="/",
        edgecolor="royalblue",
        linewidth=0.0,
        alpha=0.56,
    )
    ax.fill_betweenx(
        proton_y,
        proton_x,
        4.0,
        facecolor="red",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.02,
    )
    ax.fill_betweenx(
        proton_y,
        proton_x,
        4.0,
        facecolor="none",
        hatch="\\",
        edgecolor="red",
        linewidth=0.0,
        alpha=0.42,
    )
    ax.plot(
        photon_x,
        photon_y,
        color="blue",
        linewidth=2.3,
        label=r"$\gamma\gamma \rightarrow e^+e^-,\;\gamma\gamma \rightarrow 2(e^+e^-)$",
    )
    ax.plot(
        proton_x,
        proton_y,
        color="red",
        linewidth=2.3,
        label=r"$p\gamma \rightarrow \Delta^+ \rightarrow p\pi^0 / n\pi^+$",
    )
    ax.plot(
        iron_x,
        iron_y,
        color=IRON_COLOR,
        linewidth=2.3,
        linestyle=IRON_LINESTYLE,
        label=r"$^{56}\mathrm{Fe}+\gamma \rightarrow \mathrm{fragments}$",
    )

    add_record_lines(ax, label_x=0.0)
    ax.text(-2.0, 17.5, "photons", color="blue", fontsize=15, ha="center", bbox=LABEL_BOX)
    ax.text(0.5, 23.0, "protons", color="red", fontsize=15, ha="center", bbox=LABEL_BOX)

    ax.set_xlabel(r"$\log_{10}$(Source redshift)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\log_{10}$(particle energy / eV)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("Particle Horizon Limits", fontsize=TITLE_SIZE)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=TICK_LABEL_SIZE)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, framealpha=0.82, facecolor="white", edgecolor="none")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_curve_table(curves: dict[str, np.ndarray], output_path: Path) -> None:
    '''
    Write computed horizon curves to CSV.
    '''
    rows = ["species,process,energy_eV,horizon_redshift,horizon_distance_Mpc,static_length_Mpc"]
    for energy, redshift, distance, static_distance in zip(
        curves["photon_energy_ev"],
        curves["photon_z"],
        curves["photon_distance_mpc"],
        curves["photon_static_distance_mpc"],
    ):
        rows.append(f"photon,total_absorption,{energy:.8e},{redshift:.8e},{distance:.8e},{static_distance:.8e}")
    for energy, redshift, distance, static_distance in zip(
        curves["proton_energy_ev"],
        curves["proton_z"],
        curves["proton_distance_mpc"],
        curves["proton_static_distance_mpc"],
    ):
        rows.append(f"proton,total_horizon,{energy:.8e},{redshift:.8e},{distance:.8e},{static_distance:.8e}")
    for energy, redshift, distance, static_distance in zip(
        curves["iron_energy_ev"],
        curves["iron_z"],
        curves["iron_distance_mpc"],
        curves["iron_static_distance_mpc"],
    ):
        rows.append(f"iron,photodisintegration_elastic,{energy:.8e},{redshift:.8e},{distance:.8e},{static_distance:.8e}")
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    curves = compute_curves()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("plotting distance figure", flush=True)
    plot_mpc(curves, OUTPUT_DIR / "cosmic_limits_mpc.png")
    print("plotting redshift figure", flush=True)
    plot_redshift(curves, OUTPUT_DIR / "cosmic_limits_redshift.png")
    print("writing curve table", flush=True)
    write_curve_table(curves, OUTPUT_DIR / "computed_horizons.csv")


if __name__ == "__main__":
    main()
