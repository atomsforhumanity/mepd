"""Implicit solvent for any engine.

Two ways to put a reaction in solvent:

* **native**: the engine's own method has a solvent model (GFN2-xTB with
  ALPB/GBSA/CPCM-X). The solvent flag is simply passed through.
* **additive correction** (every other engine: g-xTB, MLIPs, ASE
  calculators): the solvation free energy of each geometry is taken from
  GFN2-xTB with the implicit model,

      E_solv(x) = E_engine(x) + [E_GFN2+model(x) - E_GFN2(x)],

  and likewise for the gradient, so geometry, TS and IRC optimizations run
  on the solvated surface too. This is a composite model: the gas-phase
  energy is the engine's, only the solvation free energy is GFN2's.

g-xTB itself has no ALPB/GBSA parameters (xtb stops with "No ALPB/GBSA
parameters found for the method/solvent"), and with `--cpcmx` it silently
returns the gas-phase energy. `SolvationCorrection` therefore checks that
the model changes the energy of the first structure it sees, and refuses
to go on if it does not.

Where it runs (mepd#40): with axtb installed, g-xTB on the axtb backend
computes the whole composite in one in-process call, with analytic
Hessians (`GXTBCalculator.axtb_solvation`, no SolvatedEngine), and the
correction for other engines runs its GFN2-xTB pair in axtb too. Both
follow axtb's conventions: by default its own (an SCF that fails with
Broyden mixing tries DIIS before smearing, so stretched structures stay
on the 300 K surface more often), "gxtb" / "xtb-6.7.1" reproduce the xtb
programs. CPCM-X, an explicit `executable`, or `backend = "xtb"` in the
[solvation] table run the xtb program as before.

Energies in solvent are E + dG_solv (ALPB's reference state: 1 M in the
gas phase to 1 M in solution, xtb's default `gsolv`). They are not full
free energies: no thermal or entropic corrections are added.

References: GFN2-xTB, C. Bannwarth, S. Ehlert, S. Grimme, JCTC 15, 1652
(2019), doi:10.1021/acs.jctc.8b01176; ALPB, S. Ehlert, M. Stahn,
S. Spicher, S. Grimme, JCTC 17, 4250 (2021), doi:10.1021/acs.jctc.1c00471;
GBSA in xtb, S. Grimme, C. Bannwarth, P. Shushkov, JCTC 13, 1989 (2017),
doi:10.1021/acs.jctc.7b00118; CPCM-X, M. Stahn, S. Ehlert, S. Grimme,
J. Phys. Chem. A 127, 7036 (2023), doi:10.1021/acs.jpca.3c04382. The
additive correction for other engines is mepd's own composite.
"""
from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from mepd.engines.engine import Engine
from mepd.engines.modified import ModifiedEngine, fresh as _fresh
from mepd.errors import ElectronicStructureError
from mepd.nodes.node import StructureNode
from mepd.cpus import available_cpus


@dataclass(frozen=True)
class Solvent:
    key: str            # xtb's name (--alpb <key>)
    label: str
    kind: str           # "protic" | "polar aprotic" | "moderately polar aprotic" | "nonpolar"
    epsilon: float      # static dielectric constant, 25 °C
    bp_c: float         # boiling point, °C at 1 atm
    mp_c: float         # melting point, °C
    models: tuple = ("alpb", "gbsa", "cpcmx")
    cpcmx_name: Optional[str] = None   # CPCM-X's own name, where it differs from `key`

    def flag(self, model: str) -> list[str]:
        """xtb's flag for this solvent in `model`: CPCM-X names solvents in full
        (--cpcmx tetrahydrofuran); with --cpcmx thf, xtb 6.7.1 stops after the SCF."""
        return [f"--{model}", (self.cpcmx_name or self.key) if model == "cpcmx" else self.key]


# Dielectric constants and phase-transition temperatures: CRC Handbook of
# Chemistry and Physics (rounded). Only solvents that GFN2-xTB's ALPB model
# is parametrized for are listed; GBSA and CPCM-X cover fewer of them
# (each model checked with xtb 6.7.1: methanol, acetone and dioxane are not in
# CPCM-X's solvent database).
SOLVENTS: dict[str, Solvent] = {s.key: s for s in [
    Solvent("water", "Water", "protic", 78.4, 100.0, 0.0),
    Solvent("methanol", "Methanol", "protic", 32.7, 64.7, -97.6, ("alpb", "gbsa")),
    Solvent("octanol", "1-Octanol", "protic", 10.3, 195.0, -16.0, ("alpb", "cpcmx")),
    Solvent("dmso", "DMSO", "polar aprotic", 46.7, 189.0, 19.0, cpcmx_name="dimethylsulfoxide"),
    Solvent("acetonitrile", "Acetonitrile", "polar aprotic", 37.5, 81.6, -45.0),
    Solvent("dmf", "DMF", "polar aprotic", 36.7, 153.0, -61.0, cpcmx_name="dimethylformamide"),
    Solvent("nitromethane", "Nitromethane", "polar aprotic", 35.9, 101.2, -28.6, ("alpb", "cpcmx")),
    Solvent("acetone", "Acetone", "polar aprotic", 20.7, 56.1, -94.7, ("alpb", "gbsa")),
    Solvent("ch2cl2", "Dichloromethane", "moderately polar aprotic", 8.9, 39.6, -96.7,
            cpcmx_name="methylenechloride"),
    Solvent("thf", "THF", "moderately polar aprotic", 7.6, 66.0, -108.4, cpcmx_name="tetrahydrofuran"),
    Solvent("ethylacetate", "Ethyl acetate", "moderately polar aprotic", 6.0, 77.1, -83.6, ("alpb", "cpcmx")),
    Solvent("chcl3", "Chloroform", "moderately polar aprotic", 4.8, 61.2, -63.5),
    Solvent("ether", "Diethyl ether", "nonpolar", 4.3, 34.6, -116.3, cpcmx_name="diethylether"),
    Solvent("toluene", "Toluene", "nonpolar", 2.4, 110.6, -95.0),
    Solvent("benzene", "Benzene", "nonpolar", 2.3, 80.1, 5.5),
    Solvent("dioxane", "1,4-Dioxane", "nonpolar", 2.2, 101.1, 11.8, ("alpb",)),
    Solvent("hexane", "n-Hexane", "nonpolar", 1.9, 68.7, -95.3),
]}

MODELS = {"alpb": "ALPB", "gbsa": "GBSA", "cpcmx": "CPCM-X"}

# xtb's CPCM-X is a correction after the SCF: it changes the energy but not
# the gradient (xtb 6.7.1, --grad --cpcmx water on HCN: gradient norm equal to
# the gas phase's to 1e-8 Eh/bohr, energy 5.6 kcal/mol lower). Optimizations,
# TS searches, IRCs and paths on it would follow gas-phase forces on a
# solvated energy, so it is for single points on given geometries only.
CPCMX_NO_GRADIENT = ("CPCM-X has no gradient in xtb (it corrects the energy after the SCF), so it can't drive "
                     "optimizations, TS searches or paths: use ALPB or GBSA for those, and CPCM-X for single points "
                     "(`mepd solvent --model cpcmx`, Keep gas-phase geometries)")


def get_solvent(name: str) -> Solvent:
    key = str(name).strip().lower()
    aliases = {"dcm": "ch2cl2", "mecn": "acetonitrile", "meoh": "methanol", "etoac": "ethylacetate",
               "et2o": "ether", "chloroform": "chcl3", "dichloromethane": "ch2cl2", "h2o": "water"}
    key = aliases.get(key, key)
    if key not in SOLVENTS:
        raise ValueError(f"unknown solvent {name!r}; known: {', '.join(SOLVENTS)}")
    return SOLVENTS[key]


def describe(solvent: str, model: str = "alpb") -> str:
    return f"{get_solvent(solvent).label} ({MODELS.get(model, model)})"


def _xtb_executable(explicit: Optional[str] = None) -> str:
    """An xtb that can run GFN2 (not the g-xTB-only `gxtb` binary):
    `explicit`, else mepd.programs.xtb_executable() ($XTB_EXECUTABLE, `xtb`
    on PATH, else the g-xTB build, downloaded on first use; it runs GFN2 +
    ALPB unless given --gxtb)."""
    from mepd.programs import standalone_gxtb, xtb_executable

    if explicit and not standalone_gxtb(explicit) and (Path(explicit).exists() or shutil.which(explicit)):
        return str(explicit)
    try:
        cand = xtb_executable()
    except Exception:
        cand = None
    if cand and not standalone_gxtb(cand):
        return str(cand)
    raise ElectronicStructureError(
        msg="Implicit solvent needs an xtb executable that runs GFN2-xTB (set XTB_EXECUTABLE, or put `xtb` "
            "on PATH). The g-xTB-only `gxtb` binary has no solvent model.")


def axtb_available() -> bool:
    from mepd.engines import axtb_backend

    return axtb_backend.version() is not None


@dataclass
class SolvationCorrection:
    """dG_solv(x) and its gradient from GFN2-xTB with an implicit model:
    the difference of two GFN2 calculations on the same geometry.

    `backend`: "axtb" (both GFN2 calculations in this process, mepd#40) or
    "xtb" (two xtb runs); None: axtb when it is installed and can do it
    (ALPB/GBSA, no explicit `executable`). `conventions`: axtb's for the
    pair (None: its default; "xtb-6.7.1": the xtb program's numbers)."""

    solvent: str
    model: str = "alpb"
    executable: Optional[str] = None
    n_parallel: int = 0
    backend: Optional[str] = None
    conventions: Optional[str] = None
    _checked: bool = field(default=False, init=False, repr=False)
    # (index in its batch, xtb flags) of every node that needed an SCC fallback
    fallbacks: list = field(default_factory=list, init=False, repr=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    # GSM pickles its engine for the gradient helper; a lock cannot be pickled.
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __post_init__(self):
        from mepd.engines.gxtb import GXTBCalculator

        sv = get_solvent(self.solvent)
        self.solvent = sv.key
        self.model = str(self.model).lower()
        if self.model not in MODELS:
            raise ValueError(f"unknown solvent model {self.model!r}; use one of {', '.join(MODELS)}")
        if self.model not in sv.models:
            raise ValueError(f"{sv.label} has no {MODELS[self.model]} parameters; use "
                             f"{' or '.join(MODELS[m] for m in sv.models)}")
        if self.backend is None:
            self.backend = "axtb" if (self.model in ("alpb", "gbsa") and not self.executable
                                      and axtb_available()) else "xtb"
        if self.backend not in ("xtb", "axtb"):
            raise ValueError(f"solvation backend must be \"xtb\" or \"axtb\", not {self.backend!r}")
        if self.backend == "axtb":
            if self.model not in ("alpb", "gbsa"):
                raise ValueError(f"axtb has no {MODELS[self.model]}: use backend = \"xtb\" (the xtb program)")
            return
        exe = _xtb_executable(self.executable)
        common = dict(executable=exe, add_gxtb_flag=False, n_parallel=self.n_parallel)
        self._gas = GXTBCalculator(**common)
        self._solv = GXTBCalculator(**common, extra_args=sv.flag(self.model))

    # Retried in this order when GFN2-xTB's SCC does not converge on a
    # geometry (e.g. stretched bonds in an anion); a node's gas and solvent
    # runs always use the same settings, so dG_solv stays a clean difference.
    # Electronic smearing comes before more iterations alone: an SCC that
    # fails in 250 iterations usually oscillates, and 1000 more then take
    # ~25 s per call only to fail too (3 such frames made a 139-frame
    # single-point solvent run take 26 s instead of 1.5 s). 5000 K last: a
    # 42-atom image in ALPB water converged only there (GBSA everywhere).
    SCC_FALLBACKS = ((), ("--iterations", "1000", "--etemp", "1000"), ("--iterations", "1000"),
                     ("--iterations", "1000", "--etemp", "5000"))

    def compute(self, nodes: list[StructureNode]) -> tuple[NDArray, NDArray]:
        """(dG_solv per node in Hartree, gradient of dG_solv in Hartree/Bohr)."""
        if not nodes:
            return np.zeros(0), np.zeros((0,))
        if self.backend == "axtb":
            dg, dgrad = self._axtb(nodes)
            self._check(dg)
            return dg, dgrad
        try:
            dg, dgrad = self._pair(self._gas, self._solv, nodes)
        except ElectronicStructureError:
            # One bad geometry fails the whole batch: redo node by node.
            from concurrent.futures import ThreadPoolExecutor

            width = max(1, min(len(nodes), self.n_parallel or available_cpus()))
            with ThreadPoolExecutor(max_workers=width) as pool:
                out = list(pool.map(lambda item: self._robust(item[1], item[0]), enumerate(nodes)))
            dg = np.array([d for d, _ in out])
            dgrad = np.array([g for _, g in out])
        self._check(dg)
        return dg, dgrad

    def _check(self, dg) -> None:
        with self._lock:
            if not self._checked:
                if not np.all(np.isfinite(dg)) or abs(float(dg[0])) < 1e-9:
                    raise ElectronicStructureError(
                        msg=f"The {MODELS[self.model]} model for {self.solvent} did not change the GFN2-xTB energy "
                            f"(dG_solv = {float(dg[0]):.3g} Eh): the xtb build ignored the solvent. Refusing to report "
                            "gas-phase energies as solvated.")
                self._checked = True

    @classmethod
    def _axtb_ladder(cls) -> list[dict]:
        """SCC_FALLBACKS as axtb Calculator settings (the same rungs, in the same order)."""
        keys = {"--iterations": ("max_iter", int), "--etemp": ("etemp", float)}
        out = []
        for flags in cls.SCC_FALLBACKS:
            rung = {}
            for name, value in zip(flags[::2], flags[1::2]):
                key, conv = keys[name]
                rung[key] = conv(value)
            out.append(rung)
        return out

    def _axtb(self, nodes: list[StructureNode]) -> tuple[NDArray, NDArray]:
        """The pair in axtb, node by node on up to n_parallel threads (axtb lets go of the GIL)."""
        from concurrent.futures import ThreadPoolExecutor

        from mepd.engines import axtb_backend

        ladder = self._axtb_ladder()

        def one(item):
            index, node = item
            dg, grad, rung = axtb_backend.solvation_pair(node, solvent=self.solvent, model=self.model, ladder=ladder,
                                                         conventions=self.conventions)
            if rung:
                with self._lock:
                    self.fallbacks.append((index, " ".join(self.SCC_FALLBACKS[rung])))
            return dg, grad

        width = max(1, min(len(nodes), self.n_parallel or available_cpus()))
        if width == 1:
            out = [one(item) for item in enumerate(nodes)]
        else:
            with ThreadPoolExecutor(max_workers=width) as pool:
                out = list(pool.map(one, enumerate(nodes)))
        return np.array([d for d, _ in out]), np.array([g for _, g in out])

    @staticmethod
    def _pair(gas_engine, solv_engine, nodes) -> tuple[NDArray, NDArray]:
        gas = [_fresh(n) for n in nodes]
        solv = [_fresh(n) for n in nodes]
        g_gas = np.asarray(gas_engine.compute_gradients(gas), dtype=float)
        g_solv = np.asarray(solv_engine.compute_gradients(solv), dtype=float)
        e_gas = np.array([n.energy for n in gas], dtype=float)
        e_solv = np.array([n.energy for n in solv], dtype=float)
        return e_solv - e_gas, g_solv - g_gas

    def _robust(self, node: StructureNode, index: int) -> tuple[float, NDArray]:
        import dataclasses

        last = None
        for extra in self.SCC_FALLBACKS:
            gas = dataclasses.replace(self._gas, extra_args=[*self._gas.extra_args, *extra], n_parallel=1)
            solv = dataclasses.replace(self._solv, extra_args=[*self._solv.extra_args, *extra], n_parallel=1)
            try:
                dg, dgrad = self._pair(gas, solv, [node])
            except ElectronicStructureError as exc:
                last = exc
                continue
            if extra:
                with self._lock:
                    self.fallbacks.append((index, " ".join(extra)))
            return float(dg[0]), dgrad[0]
        raise ElectronicStructureError(
            msg=f"GFN2-xTB ({MODELS[self.model]}, {self.solvent}) did not converge for structure {index} of this "
                f"batch, even with {' '.join(self.SCC_FALLBACKS[-1])}: no solvation energy for it.",
            obj=getattr(last, "obj", None))


@dataclass
class SolvatedEngine(ModifiedEngine):
    """`base` on the solvated surface: E = E_base + dG_solv, with gradients,
    Hessians, geometry/TS optimizations and IRCs all on that surface."""

    base: Engine
    solvent: str
    model: str = "alpb"
    method: str = "correction"     # "correction" | "native"
    executable: Optional[str] = None
    backend: Optional[str] = None  # the correction's: "axtb" | "xtb" | None (see SolvationCorrection)

    def __post_init__(self):
        self.solvent = get_solvent(self.solvent).key
        self.correction = None if self.method == "native" else SolvationCorrection(
            self.solvent, self.model, executable=self.executable,
            n_parallel=int(getattr(self.base, "n_parallel", 0) or 0), backend=self.backend)
        self._inherit()

    @property
    def label(self) -> str:
        how = "native" if self.method == "native" else "GFN2-xTB solvation correction"
        return f"{describe(self.solvent, self.model)}, {how}"

    def _term(self, nodes):
        return None if self.correction is None else self.correction.compute(nodes)


def uses_correction(engine: Engine) -> bool:
    """Whether `engine` (from solvate_engine) is in solvent through the GFN2-xTB correction, the
    composite (a SolvatedEngine, or g-xTB with axtb's composite), not a solvent model of its own."""
    from mepd.engines.gxtb import GXTBCalculator

    for _ in range(8):
        if isinstance(engine, SolvatedEngine):
            return engine.method != "native"
        if isinstance(engine, GXTBCalculator):
            return bool(engine.axtb_solvation)
        inner = getattr(engine, "base", None)
        if inner is None or inner is engine:
            return False
        engine = inner
    return False


def _axtb_composite_capable(engine: Engine) -> bool:
    """g-xTB on the axtb backend: axtb computes the whole composite in one call (mepd#40)."""
    from mepd.engines.gxtb import GXTBCalculator

    return isinstance(engine, GXTBCalculator) and engine.backend == "axtb" and engine.add_gxtb_flag \
        and not engine.axtb_solvation


def _native_capable(engine: Engine) -> bool:
    """GFN2-xTB run through the local xtb engine: its own ALPB/GBSA/CPCM-X."""
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.programs import standalone_gxtb

    return isinstance(engine, GXTBCalculator) and not engine.add_gxtb_flag and not standalone_gxtb(engine.executable)


#: our solvent keys -> PCMSolver's names (Psi4's PCM)
PCM_SOLVENTS = {"water": "Water", "methanol": "Methanol", "dmso": "DMSO", "acetonitrile": "Acetonitrile",
                "nitromethane": "Nitromethane", "acetone": "Acetone", "ch2cl2": "Methylenechloride",
                "thf": "THF", "chcl3": "Chloroform", "toluene": "Toluene", "benzene": "Benzene",
                "dioxane": "1,4-Dioxane"}


#: our solvent keys -> ORCA's CPCM names
ORCA_SOLVENTS = {"water": "water", "methanol": "methanol", "octanol": "octanol", "dmso": "dmso",
                 "acetonitrile": "acetonitrile", "dmf": "dmf", "nitromethane": "nitromethane", "acetone": "acetone",
                 "ch2cl2": "ch2cl2", "thf": "thf", "ethylacetate": "ethylacetate", "chcl3": "chloroform",
                 "ether": "diethylether", "toluene": "toluene", "benzene": "benzene", "dioxane": "1,4-dioxane",
                 "hexane": "hexane"}


def solvate_engine(engine: Engine, solvation: dict) -> Engine:
    """`engine` in the solvent that a profile's `[solvation]` table asks
    for: {solvent, model = "alpb", method = "auto" | "native" | "correction",
    backend = "axtb" | "xtb" (where the correction's GFN2-xTB runs; default
    axtb when installed), executable (an xtb for it; implies "xtb")}.
    An engine with its own implicit solvent (GFN2-xTB's ALPB/GBSA, Psi4's
    PCM, ORCA's CPCM) uses it: "auto" never adds the xTB correction on top of one.
    g-xTB on the axtb backend gets the composite from axtb in one call."""
    from mepd.engines.orca import OrcaEngine
    from mepd.engines.psi4 import Psi4Engine

    sv = get_solvent(solvation["solvent"])
    solvent = sv.key
    model = str(solvation.get("model", "alpb")).lower()
    method = str(solvation.get("method", "auto")).lower()
    if model not in MODELS:
        raise ValueError(f"unknown solvent model {model!r}; use one of {', '.join(MODELS)}")
    if model == "cpcmx":
        raise ValueError(CPCMX_NO_GRADIENT)
    if model not in sv.models:
        raise ValueError(f"{sv.label} has no {MODELS[model]} parameters; use "
                         f"{' or '.join(MODELS[m] for m in sv.models)}")
    if method not in ("auto", "native", "correction"):
        raise ValueError("solvation.method must be auto, native or correction")
    if isinstance(engine, Psi4Engine) and method in ("auto", "native"):
        import dataclasses

        if solvent not in PCM_SOLVENTS:
            raise ValueError(f"Psi4's PCM has no {solvent} here (it has {', '.join(PCM_SOLVENTS)}); "
                             "set method = \"correction\" for the GFN2-xTB solvation correction instead")
        return dataclasses.replace(engine, pcm_solvent=PCM_SOLVENTS[solvent])
    if isinstance(engine, OrcaEngine) and method in ("auto", "native"):
        import dataclasses

        if solvent not in ORCA_SOLVENTS:
            raise ValueError(f"ORCA's CPCM has no {solvent} here (it has {', '.join(ORCA_SOLVENTS)}); "
                             "set method = \"correction\" for the GFN2-xTB solvation correction instead")
        return dataclasses.replace(engine, cpcm_solvent=ORCA_SOLVENTS[solvent])
    native = _native_capable(engine)
    if method == "native" and not native:
        raise ValueError(f"{type(engine).__name__} has no native implicit solvent here; use method = \"correction\"")
    if method in ("auto", "native") and native:
        import dataclasses

        return dataclasses.replace(engine, extra_args=[*engine.extra_args, *sv.flag(model)])
    backend = solvation.get("backend")
    if backend is not None:
        backend = str(backend).lower()
        if backend not in ("xtb", "axtb"):
            raise ValueError(f"solvation.backend must be \"axtb\" or \"xtb\", not {backend!r}")
    if _axtb_composite_capable(engine) and backend != "xtb" and not solvation.get("executable"):
        import dataclasses

        return dataclasses.replace(engine, axtb_solvation={"solvent": solvent, "model": model})
    return SolvatedEngine(engine, solvent, model=model, method="correction",
                          executable=solvation.get("executable"), backend=backend)
