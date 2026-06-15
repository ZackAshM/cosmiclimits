# Cosmic Particle Horizon from Background Interactions

<table>
  <tr>
    <td><img src="output/cosmic_limits_mpc.png" alt="Particle horizon limits by observable distance" width="100%"></td>
    <td><img src="output/cosmic_limits_redshift.png" alt="Particle horizon limits by source redshift" width="100%"></td>
  </tr>
</table>

This package creates cosmic propagation limits, or horizons, plots for particles interacting with cosmological thermal backgrounds (e.g. CMB).

The physics backend for photon and cosmic ray processes is built from [CRPropa3-data](https://github.com/CRPropa/CRPropa3-data). Local code is limited to selecting CRPropa photon-background models, wrapping CRPropa interaction-rate calculators, solving the depth-equals-one horizon condition, and plotting the results.

## Defining Particle Horizons

The particle-horizon definitions used here follow Ruffini, Vereshchagin, and Xue, *Cosmic absorption of ultra high energy particles*, Astrophys. Space Sci. 361, 82 (2016), Secs. 3–4, https://arxiv.org/pdf/1503.07749. 

A particle horizon is defined by the condition that the accumulated optical depth reaches unity for a particle propagating through cosmological background radiation fields:

$$\tau = 1$$

Without cosmological redshift evolution, the horizon is equivalent to the particle’s mean free path.

$$\lambda(E) = \frac{1}{\sigma(E)\,n}$$

or, equivalently for the CRPropa-derived local interaction rates used here,

$$\lambda(E) = \frac{1}{\Gamma_0(E)}$$

where $\Gamma_0(E)$ is the local $z = 0$ interaction rate in units of `Mpc⁻¹`.

### Including Cosmological Redshift Evolution

Outlined in the paper above, we can extend the propagation to include effects of redshift, both for evolution of the traveling particle and the cosmological backgrounds. Still, the horizon is defined at

$$\tau(E_0, z_h) = 1~,$$

now with an optical-depth integral of the form

$$
\tau(E_0, z_h)
=
\int_0^{z_h}
\Gamma\!\left(E_0(1+z')^2\right)
(1+z')^3
\frac{D_H\,dz'}{(1+z')\,\mathcal{H}(z')}
$$

where $E_0$ is the particle energy observed today, $D_H = c/H_0$, and

$$\mathcal{H}(z) = [\Omega_r(1+z)^4 + \Omega_M(1+z)^3 + \Omega_\Lambda]^{1/2}$$

In this implementation, $\Gamma(E)$ is evaluated from local CRPropa rate tables and scaled by the background-density factor $(1+z)^3$; the shifted argument $E_0(1+z')^2$ reflects the combined redshifting of the propagating particle and background energy scales.

For interactions where the primary particle is not annihilated and does not lose a significant fraction of its energy in a single interaction, the mean free path is not itself the appropriate horizon scale. Ruffini et al. instead define a mean energy-loss distance,

$$
\tilde{\lambda}^{-1} = \frac{1}{E} \frac{dE}{c\,dt},
$$

and the corresponding accumulated energy-loss depth,

$$
\tilde{\tau}
=
\int_0^t
\frac{c\,dt'}{\tilde{\lambda}}
=
D_H
\int_0^z
\frac{dz'}{\tilde{\lambda}(z')\,(1+z')\,\mathcal{H}(z')}.
$$


## Contributing Physics Processes

### Photons

- Pair production: `γγ → e⁺e⁻`
- Double pair production: `γγ → 2(e⁺e⁻)`

Both photon processes remove the primary photon, so the photon horizon uses the sum of the two CRPropa-derived absorption rates.

### Protons

- Photopion production: `pγ → Δ⁺ → pπ⁰ / nπ⁺`

The neutral pion channel produces photon secondaries through `π⁰ → γγ`; the charged pion channel produces leptonic and neutrino secondaries through the usual pion/muon decay chain. This plot treats the proton curve as a photopion interaction mean-free-path horizon. Proton electron-pair-production energy losses are implemented in `proton/electron_pair_production.py` for comparison work, but they are not included in these primary plots.

## Model Choices

- CRPropa data source: local checkout of `CRPropa3-data`.
- Photon backgrounds: `CMB`, `EBL_Saldana21`, and `URB_Fixsen11`.
- Photon interaction grid: CRPropa electromagnetic cross sections sampled on a `2^18 + 1` Romberg grid, with a field-aware upper `s_kin` bound that extends beyond CRPropa's default when required by the plotted energy range.
- Proton photopion cross section: full shipped `tables/PPP/xs_proton.txt` table, regridded to `2^12 + 1` log-spaced samples for CRPropa's Romberg integration.
- Particle energy ranges: photons use `10^10` to `10^25 eV`; protons use `10^17` to `10^25 eV`.
- Horizon integration range: redshift integration uses `z_max = 40`.

## Install

Clone the repository with its CRPropa3-data submodule:

```bash
git clone --recurse-submodules <repo-url>
cd cosmiclimits
```

If the repository was cloned without submodules, initialize them from the project root:

```bash
git submodule update --init --recursive
```

Create the local virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

The compiled CRPropa Python package is not required for these plots. This project uses the local `data/CRPropa3-data` submodule as the physics-data backend.

## Usage

Run the plotter script from the project root:

```bash
.venv/bin/python scripts/plot.py
```

Plot results are written inside `output/`:

- `cosmic_limits_mpc.png`
- `cosmic_limits_redshift.png`
- `computed_horizons.csv`

Modules:

- `data/crpropa_runtime.py`: registers the local CRPropa3-data checkout and imports CRPropa data-generation modules.
- `cosmiclimits/photon/`: wraps CRPropa photon pair-production and double-pair-production rate calculations.
- `cosmiclimits/proton/`: wraps CRPropa proton photopion and electron-pair-production calculations.
- `cosmiclimits/horizon.py`: solves the depth-equals-one horizon condition and converts redshift to observable distance.
- `cosmiclimits/utils.py`: stores/interpolates generated rate tables.
- `scripts/plot.py`: computes the curves, draws figures, and writes the CSV table.
