"""Free energies for the kinetic model: quasi-RRHO thermochemistry from
GFN2-xTB single-point Hessians.

G = E + G_corr(T), with E the energy at the workspace's level (e.g. g-xTB,
with its implicit solvent) and G_corr = ZPE + H_thermal(T) - T*S(T) from the
vibrations, rotation and translation of the structure:

  * frequencies come from a GFN2-xTB "single-point Hessian" (`xtb --bhess`),
    which is built for exactly this case: a geometry that is a stationary
    point at another level (here g-xTB) and so not quite one at GFN2-xTB.
    The biased Hessian keeps the low modes real instead of spuriously
    imaginary. The implicit solvent of the workspace level is used for the
    Hessian too;
  * modes below ~50 cm^-1 are treated as hindered rotors (Grimme's
    quasi-RRHO): their entropy is interpolated toward free-rotor entropy,
    since a harmonic oscillator gives an absurdly large entropy to a very
    soft mode;
  * a transition state's one imaginary mode is left out;
  * the standard state is 1 mol/L, matching the solvation free energies of
    the implicit solvent model (1 M gas -> 1 M solution).

What it does not do: conformer averaging (one structure per species), or
any correction to the translational entropy lost on association in
solution (gas-phase formulas overstate it).

References: quasi-RRHO, S. Grimme, Chem. Eur. J. 18, 9955 (2012),
doi:10.1002/chem.201200497; single-point Hessians, S. Spicher, S. Grimme,
J. Chem. Theory Comput. 17, 1701 (2021), doi:10.1021/acs.jctc.0c01306.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.constants import Avogadro, Boltzmann, Planck, atm, atomic_mass, c, gas_constant

KCAL = 4184.0
HARTREE_KCAL = 627.509474
ROTOR_CUTOFF_CM = 50.0      # xtb's default for the quasi-RRHO switch
B_AV = 1e-44                # kg m^2: Grimme's average moment of inertia for the free-rotor limit
METHOD = "gfn2-bhess"


def _masses(symbols: Sequence[str]) -> np.ndarray:
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    return np.array([table.GetAtomicWeight(s) for s in symbols])


def _xtb() -> str:
    """An xtb that runs GFN2-xTB (the g-xTB build does unless given --gxtb)."""
    from mepd.solvation import _xtb_executable

    return _xtb_executable()


def frequencies(symbols: Sequence[str], coords_ang: np.ndarray, *, charge: int = 0, multiplicity: int = 1,
                solvent: Optional[str] = None, model: str = "alpb", executable: Optional[str] = None) -> dict:
    """GFN2-xTB single-point Hessian of a geometry: {freqs (cm^-1, without
    the translations and rotations), rotational_number, linear, xtb_g298_eh
    (xtb's own G_RRHO at 298.15 K and 1 atm, for checking)}."""
    work = Path(tempfile.mkdtemp(prefix="mepd_thermo_"))
    try:
        lines = [str(len(symbols)), ""] + [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in
                                            zip(symbols, np.asarray(coords_ang).reshape(-1, 3))]
        (work / "in.xyz").write_text("\n".join(lines) + "\n")
        cmd = [executable or _xtb(), "in.xyz", "--bhess", "--chrg", str(int(charge)),
               "--uhf", str(int(multiplicity) - 1)]
        if solvent:
            cmd += [f"--{model}", solvent]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE", "1G")}
        run = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
        spec = work / "vibspectrum"
        if run.returncode != 0 or not spec.exists():
            raise RuntimeError(f"xtb --bhess failed (exit {run.returncode}): {(run.stdout + run.stderr)[-600:]}")
        out = run.stdout
        linear = "linear?" in out and "true" in out.split("linear?")[1].split("\n")[0]
        rot = 1
        if "rotational number" in out:
            rot = int(out.split("rotational number")[1].split()[0])
        g298 = None
        if "G(RRHO) contrib." in out:
            g298 = float(out.split("G(RRHO) contrib.")[1].split()[0])
        modes = []
        for ln in spec.read_text().splitlines():
            t = ln.split()
            if not t or t[0].startswith(("#", "$")):
                continue
            modes.append(float(t[-3]))
        n_tr = 5 if linear or len(symbols) == 2 else 6
        if len(symbols) == 1:
            n_tr = len(modes)
        # Drop the translations and rotations: the modes closest to zero.
        keep = sorted(range(len(modes)), key=lambda k: abs(modes[k]))[n_tr:]
        freqs = sorted(modes[k] for k in keep)
        return {"freqs": freqs, "rotational_number": rot, "linear": bool(linear), "xtb_g298_eh": g298,
                "method": METHOD, "solvent": solvent, "model": model if solvent else None}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def qrrho(freqs: Sequence[float], symbols: Sequence[str], coords_ang: np.ndarray, temperature: float, *,
          rotational_number: int = 1, linear: bool = False, multiplicity: int = 1, ts: bool = False,
          rotor_cutoff: float = ROTOR_CUTOFF_CM, standard_state: str = "1M") -> dict:
    """Thermal corrections in kcal/mol: {zpe, h (ZPE + thermal enthalpy),
    s (cal/mol/K), g_corr (= h - T s), n_imaginary}. `ts`: the most
    negative mode is the reaction coordinate and is left out. Other
    imaginary modes (numerical noise) count with their absolute value.
    `standard_state`: "1M" (default) or "1atm"."""
    T = float(temperature)
    R = gas_constant
    kT = Boltzmann * T
    f = sorted(float(x) for x in freqs)
    n_imag = sum(1 for x in f if x < 0)
    if ts and f and f[0] < 0:
        f = f[1:]
    f = [abs(x) for x in f if abs(x) > 1e-6]
    zpe = 0.0
    e_vib = 0.0
    s_vib = 0.0
    for nu in f:
        hv = Planck * c * nu * 100.0                 # J
        x = hv / kT
        zpe += 0.5 * hv * Avogadro
        e_vib += hv * Avogadro / math.expm1(x)
        s_ho = R * (x / math.expm1(x) - math.log(-math.expm1(-x)))
        # Free-rotor entropy of a mode with that frequency (Grimme 2012).
        mu = Planck / (8 * math.pi ** 2 * c * nu * 100.0)
        mu_eff = mu * B_AV / (mu + B_AV)
        s_fr = R * (0.5 + math.log(math.sqrt(8 * math.pi ** 3 * mu_eff * kT / Planck ** 2)))
        w = 1.0 / (1.0 + (rotor_cutoff / nu) ** 4)
        s_vib += w * s_ho + (1 - w) * s_fr
    # Rotation (rigid rotor, high-temperature limit).
    m = _masses(symbols) * atomic_mass
    xyz = np.asarray(coords_ang, dtype=float).reshape(-1, 3) * 1e-10
    n = len(symbols)
    if n == 1:
        e_rot, s_rot = 0.0, 0.0
    else:
        com = (m[:, None] * xyz).sum(0) / m.sum()
        r = xyz - com
        inertia = np.zeros((3, 3))
        for mi, ri in zip(m, r):
            inertia += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
        moments = np.sort(np.linalg.eigvalsh(inertia))
        if linear or n == 2 or moments[0] < 1e-3 * moments[2]:
            i_lin = moments[2]
            e_rot = R * T
            s_rot = R * (math.log(8 * math.pi ** 2 * i_lin * kT / (rotational_number * Planck ** 2)) + 1.0)
        else:
            e_rot = 1.5 * R * T
            q = (math.sqrt(math.pi * moments.prod()) / rotational_number
                 * (8 * math.pi ** 2 * kT / Planck ** 2) ** 1.5)
            s_rot = R * (math.log(q) + 1.5)
    # Translation (Sackur-Tetrode), at 1 mol/L or 1 atm.
    mass = m.sum()
    volume = (1.0 / (1000.0 * Avogadro)) if standard_state == "1M" else kT / atm
    q_tr = (2 * math.pi * mass * kT / Planck ** 2) ** 1.5 * volume
    s_tr = R * (math.log(q_tr) + 2.5)
    h_tr = 2.5 * R * T                               # 3/2 RT + PV
    s_el = R * math.log(max(1, int(multiplicity)))
    h = (zpe + e_vib + e_rot + h_tr) / KCAL
    s = (s_vib + s_rot + s_tr + s_el) / 4.184        # cal/mol/K
    return {"zpe": zpe / KCAL, "h": h, "s": s, "g_corr": h - T * s / 1000.0, "n_imaginary": n_imag}


def cache_key(symbols: Sequence[str], coords_ang: np.ndarray, charge: int, multiplicity: int,
              solvent: Optional[str], model: str) -> str:
    """A geometry's identity for the frequency cache (coordinates to 1e-5 Angstrom)."""
    xyz = np.round(np.asarray(coords_ang, dtype=float).reshape(-1, 3), 5) + 0.0
    blob = json.dumps([list(symbols), xyz.tolist(), int(charge), int(multiplicity), solvent, model, METHOD])
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def cached_frequencies(cache_dir: Path, symbols, coords_ang, *, charge=0, multiplicity=1, solvent=None,
                       model="alpb") -> dict:
    """frequencies(), computed once per geometry and kept in `cache_dir`."""
    key = cache_key(symbols, coords_ang, charge, multiplicity, solvent, model)
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    out = frequencies(symbols, coords_ang, charge=charge, multiplicity=multiplicity, solvent=solvent, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return out
