"""ORCA as an mepd engine: its own optimizers and analytic Hessians.

ORCA (free for academics, licensed; https://www.faccts.de/orca/) is the
fastest CPU code we have for DFT on molecules: RIJCOSX makes hybrid
functionals with triple-zeta basis sets cheap, its optimizer works in
internal coordinates, and its Hessians are analytic. So minimizations, TS
searches and Hessians run inside ORCA (Opt, OptTS with an analytic starting
Hessian, Freq), not through mepd's generic optimizers with finite
differences. The IRC is mepd's usual one (Sella) on ORCA gradients, so it
follows the same rules as every other engine.

Inputs are written and outputs read by qccodec's ORCA codec. ORCA is run by
its full path (its parallel runs need that). The executable: `executable`,
$MEPD_ORCA, /opt/orca/orca, or `orca` on PATH -- only when ORCA's own
programs (orca_scf or orca_scfgrad, ...) sit beside it, since `orca` is also the name of
GNOME's screen reader.

`threads` is ORCA's %pal nprocs (MPI processes) and `maxcore` its memory per
process in MB, so a calculation needs about threads x maxcore of memory.

ORCA: F. Neese, WIREs Comput. Mol. Sci. 12, e1606 (2022), doi:10.1002/wcms.1606.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.errors import ConvergenceError, ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache

INSTALL = ("ORCA was not found: install it (https://www.faccts.de/orca/, free for academics) and set "
           "MEPD_ORCA to its `orca` executable, or pass `executable`")
#: Eh/bohr per eV/Angstrom
_EV_PER_A_TO_EH_PER_BOHR = 0.529177210903 / 27.211386245988
_HESSIANS: dict = {}   # OrcaEngine.compute_hessian, per process
#: geometry steps when none are asked for (ORCA's own default is too few for a
#: loose pair of molecules, e.g. an IRC end, at TightOpt)
_MAXITER = {"minimization": 300, "TS search": 200}


def orca_executable(explicit: Optional[str] = None) -> Optional[str]:
    """The ORCA program, or None. A candidate counts only with ORCA's own
    programs (orca_scf*) beside it (`orca` on PATH may be GNOME's screen reader)."""
    cands = [explicit, os.getenv("MEPD_ORCA"), "/opt/orca/orca", shutil.which("orca")]
    for c in cands:
        if c and Path(c).is_file() and any(Path(c).resolve().parent.glob("orca_scf*")):
            return str(Path(c).resolve())
    return None


@dataclass
class OrcaEngine(Engine):
    """`method`/`basis`: ORCA's names (e.g. "wB97X-D3BJ", "def2-TZVP");
    `keywords`: more words for the ! line (default RIJCOSX with def2/J,
    DefGrid2, TightSCF); `blocks`: extra %blocks, e.g. {"scf": {"maxiter": 300}};
    `threads`: %pal nprocs; `maxcore`: MB per process; `cpcm_solvent`:
    ORCA's CPCM solvent name (set by a profile's [solvation] table);
    `opt_convergence`: ORCA's convergence keyword for minimizations and TS
    searches; `ts_hessian_every`: OptTS recomputes the analytic Hessian every
    this many steps (0: only at the start)."""

    method: str = "wB97X-D3BJ"
    basis: str = "def2-TZVP"
    keywords: str = "def2/J RIJCOSX DefGrid2 TightSCF"
    blocks: dict = field(default_factory=lambda: {"scf": {"maxiter": 300}})
    threads: int = 8
    maxcore: int = 2000
    cpcm_solvent: Optional[str] = None
    opt_convergence: str = "TightOpt"
    ts_hessian_every: int = 10
    executable: Optional[str] = None
    timeout: float = 86400.0
    keep_workdirs: bool = False

    def __post_init__(self):
        self.executable = orca_executable(self.executable)
        if not self.executable:
            raise ElectronicStructureError(msg=INSTALL)

    # ------------------------------------------------------------ running ORCA
    def _model_words(self, extra: str = "") -> str:
        words = [self.method, self.keywords]
        if self.cpcm_solvent:
            words.append(f"CPCM({self.cpcm_solvent})")
        if extra:
            words.append(extra)
        return " ".join(w for w in words if w)

    def _run(self, structure, calctype, *, extra_words: str = "", blocks: Optional[dict] = None):
        """qccodec's decoded data and ORCA's stdout for one calculation."""
        import qccodec
        from qcdata import Model, ProgramInput

        kw = {"pal": int(self.threads), "maxcore": int(self.maxcore)}
        for name, block in {**self.blocks, **(blocks or {})}.items():
            kw[name] = {**dict(self.blocks.get(name, {})), **dict(block)}
        inp = ProgramInput(program="orca", calctype=calctype, structure=structure,
                           model=Model(method=self._model_words(extra_words), basis=self.basis), keywords=kw)
        native = qccodec.encode(inp)
        with tempfile.TemporaryDirectory(prefix="mepd-orca-") as tmp:
            wd = Path(tmp)
            (wd / "orca.inp").write_text(native.input_file)
            (wd / native.geometry_filename).write_text(native.geometry_file)
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            env["PATH"] = f"{Path(self.executable).parent}{os.pathsep}{env.get('PATH', '')}"
            try:
                proc = subprocess.run([self.executable, "orca.inp"], cwd=wd, env=env, capture_output=True,
                                      text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise ElectronicStructureError(msg=f"ORCA timed out after {self.timeout:g} s") from exc
            stdout = proc.stdout
            ok = "ORCA TERMINATED NORMALLY" in stdout
            if self.keep_workdirs:
                keep = Path.cwd() / "orca-workdirs"
                keep.mkdir(exist_ok=True)
                shutil.copytree(wd, keep / wd.name, dirs_exist_ok=True)
            try:
                data = qccodec.decode("orca", calctype, stdout=stdout, directory=wd, input_data=inp, failed=not ok)
            except Exception as exc:
                if ok:
                    raise ElectronicStructureError(msg=f"could not read ORCA's output: {exc}", obj=stdout) from exc
                data = None
        return data, stdout, ok

    def _single_point(self, structure, calctype):
        data, stdout, ok = self._run(structure, calctype)
        if not ok or data is None:
            raise ElectronicStructureError(msg="ORCA failed:\n" + (stdout[-3000:] or "no output"), obj=stdout)
        return data

    # ------------------------------------------------------------ engine
    def _compute(self, chain: Union[Chain, List], gradient: bool = True) -> list:
        from qcdata import CalcType

        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or (gradient and n._cached_gradient is None)]
        if not gradient:
            for n in todo:
                n._cached_energy = float(self._single_point(n.structure, CalcType.energy).energy)
            return nodes
        results = []
        for n in todo:
            d = self._single_point(n.structure, CalcType.gradient)
            results.append(FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                {"energy": float(d.energy), "gradient": np.asarray(d.gradient, dtype=float)})}))
        if todo:
            update_node_cache(node_list=todo, results=results)
        return nodes

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.gradient for n in self._compute(chain)])

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.energy for n in self._compute(chain, gradient=False)])

    def compute_hessian(self, node, step_size: float | None = None) -> NDArray:
        """ORCA's analytic Hessian (Eh/bohr^2); `step_size` is not used. Kept
        per geometry, so a frequency check and a thermal correction at the
        same structure share one."""
        from qcdata import CalcType

        key = (self.method, self.basis, self.keywords, self.cpcm_solvent, tuple(node.symbols),
               int(node.structure.charge), int(node.structure.multiplicity),
               np.round(np.asarray(node.coords, dtype=float), 8).tobytes())
        if key not in _HESSIANS:
            _HESSIANS[key] = np.asarray(self._single_point(node.structure, CalcType.hessian).hessian, dtype=float)
        return _HESSIANS[key].copy()

    def _optimize(self, node, calctype, keywords: Optional[dict], geom: dict) -> list[StructureNode]:
        kw = dict(keywords or {})
        maxiter = kw.pop("maxiter", kw.pop("maxit", kw.pop("steps", None)))
        what = "TS search" if "calc_hess" in geom else "minimization"
        geom["maxiter"] = int(maxiter) if maxiter is not None else _MAXITER[what]
        fmax = kw.pop("fmax", None)
        if fmax is not None:      # largest force, eV/Angstrom -> ORCA's TolMAXG (Eh/bohr)
            geom["tolmaxg"] = float(fmax) * _EV_PER_A_TO_EH_PER_BOHR
        data, stdout, ok = self._run(node.structure, calctype, extra_words=self.opt_convergence, blocks={"geom": geom})
        traj = list(getattr(data, "trajectory", None) or [])
        nodes = []
        for po in traj:
            n = StructureNode(structure=po.input_data.structure)
            n._cached_energy = float(po.results.energy)
            if po.results.gradient is not None:
                n._cached_gradient = np.asarray(po.results.gradient, dtype=float)
            nodes.append(n)
        converged = ok and "THE OPTIMIZATION HAS CONVERGED" in stdout
        if not converged:
            err = ConvergenceError(f"ORCA's {what} did not converge" + (f" in {geom['maxiter']} steps" if ok else ":\n" + stdout[-2000:]))
            err.structure = nodes[-1] if nodes else None
            raise err
        return nodes

    def compute_geometry_optimization(self, node, keywords: dict | None = None) -> list[StructureNode]:
        """ORCA's own minimizer (`opt_convergence`); keywords `maxiter` (default 300), and
        `fmax` (eV/Angstrom) for the largest-force criterion."""
        from qcdata import CalcType

        return self._optimize(node, CalcType.optimization, keywords, {})

    def compute_transition_state(self, node, keywords: dict | None = None) -> StructureNode:
        """ORCA's OptTS from an analytic Hessian, recomputed every
        `hessian_every` steps (keyword; default `ts_hessian_every`); `maxiter`
        (default 200)."""
        from qcdata import CalcType

        kw = dict(keywords or {})
        kw.pop("exact_hessian", None)
        every = kw.pop("hessian_every", None)
        every = self.ts_hessian_every if every is None else int(every)
        geom = {"calc_hess": True}
        if every:
            geom["recalc_hess"] = int(every)
        for k in ("optimizer_kwds", "optimizer_kwargs", "order"):
            kw.pop(k, None)
        return self._optimize(node, CalcType.transition_state, kw, geom)[-1]

    def _ase(self, node):
        from mepd.engines.ase import ASEEngine
        from mepd.engines.gxtb import _GXTBASEResultsCalculator

        return ASEEngine(calculator=_GXTBASEResultsCalculator(self, charge=int(node.structure.charge),
                                                              multiplicity=int(node.structure.multiplicity)))

    def compute_irc_chain(self, ts_node, keywords: dict | None = None):
        """mepd's IRC (Sella) on ORCA gradients."""
        return self._ase(ts_node).compute_irc_chain(ts_node=ts_node, keywords=keywords)

