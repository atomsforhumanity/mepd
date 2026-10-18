"""TeraChem as an mepd engine (`engine_name = "terachem"`), run locally.

Each energy/gradient is one `terachem tc.in` run in a fresh directory
(`command`: $MEPD_TERACHEM or `terachem` on PATH, e.g. the mtzgroup docker
wrapper). The orbitals of the previous run of the same molecule are the next
run's guess, so along a path or an optimization the SCF starts converged.

Besides plain energies and gradients it computes them in a field of point
charges (`energy_gradient(..., point_charges=...)`, TeraChem's
`pointcharges` file), with the gradient on the charges: TeraChem can be the
QM level of electrostatic QM/MM in mepd.engines.qmmm (TIP3P water, a
protein), like xtb and Psi4.

The license: TeraChem reads $TERACHEM_LICENSE_PATH (the docker wrapper mounts
it); when that is unset and ~/.config/terachem/license.key exists, mepd uses
that file.

TeraChem: I. S. Ufimtsev and T. J. Martinez, J. Chem. Theory Comput. 5, 2619
(2009), doi:10.1021/ct9003004; S. Seritan et al., WIREs Comput. Mol. Sci. 11,
e1494 (2021), doi:10.1002/wcms.1494.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from qcref.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.nodehelpers import update_node_cache

LICENSE_FILE = Path.home() / ".config" / "terachem" / "license.key"

INSTALL = ("TeraChem was not found: put `terachem` on PATH (a native install, or the docker wrapper for "
           "mtzgroup/terachem images) or set $MEPD_TERACHEM to the command, and keep the license in "
           f"$TERACHEM_LICENSE_PATH or {LICENSE_FILE}.")


def ensure_license_env() -> Optional[str]:
    """$TERACHEM_LICENSE_PATH, set to ~/.config/terachem/license.key when
    unset and that file exists (what the docker wrapper mounts)."""
    if not os.environ.get("TERACHEM_LICENSE_PATH") and LICENSE_FILE.is_file():
        os.environ["TERACHEM_LICENSE_PATH"] = str(LICENSE_FILE)
    return os.environ.get("TERACHEM_LICENSE_PATH")


def terachem_command(command: Optional[str] = None) -> Optional[list[str]]:
    cmd = command or os.environ.get("MEPD_TERACHEM") or "terachem"
    parts = shlex.split(cmd)
    return parts if parts and shutil.which(parts[0]) else None


# ------------------------------------------------------------- output text
def parse_gradients(text: str) -> tuple[np.ndarray, np.ndarray]:
    """QM-atom gradient and the point-charge gradient (Eh/bohr) from
    TeraChem's stdout: the dE/dX block, then '--- MM / Point charge part ---'."""
    from mepd.engines.terachem_qmmm import parse_qmmm_gradients

    return parse_qmmm_gradients(text)


def parse_energy(text: str) -> float:
    from mepd.engines.terachem_qmmm import parse_final_energy

    return parse_final_energy(text)


def parse_mulliken(path: Path) -> Optional[np.ndarray]:
    """Atomic charges from scr/charge_mull.xls (index, element, charge, ...)."""
    try:
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        return np.array([float(r[2]) for r in rows if len(r) >= 3 and r[0].isdigit()])
    except (OSError, ValueError, IndexError):
        return None


def _failure(text: str) -> str:
    """The lines that say why a TeraChem run gave no result."""
    sms = re.search(r"Supported architecture SMs:\s*([\d ]+)", text)
    cc = re.search(r"Device \d+: ([^,]+),.*?CC (\d+)\.(\d+)", text)
    if sms and cc:
        have = {int(v) for v in sms.group(1).split()}
        gpu = int(cc.group(2)) * 10 + int(cc.group(3))
        if gpu not in have:
            return (f"this TeraChem build has GPU code for SM {', '.join(map(str, sorted(have)))} only; "
                    f"{cc.group(1).strip()} is SM {gpu // 10}.{gpu % 10}. Use a TeraChem build for this GPU.")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bad = [line for line in lines if re.search(r"error|cannot|can't|failed|not converge|abort", line, re.I)
           and "license.dat" not in line]
    bad += [line for line in lines if re.search(r"licen[cs]e", line, re.I)
            and re.search(r"invalid|expired|denied|no (valid )?licen|not (be )?(found|available)|unable", line, re.I)]
    return "\n".join((bad or lines)[-6:])


@dataclass
class TeraChemEngine(Engine):
    """`method`/`basis` (TeraChem names: b3lyp, wpbeh, hf, ...; 6-31gs,
    def2-svp, ...), `keywords`: any other tc.in lines (e.g. dftd = "d3",
    convthre = 1e-6), `command`: how to start TeraChem, `timeout` (s) per
    run, `keep_runs`: keep each run's directory (under `workdir`)."""

    method: str = "b3lyp"
    basis: str = "6-31gs"
    keywords: dict = field(default_factory=dict)
    command: Optional[str] = None
    timeout: float = 3600.0
    workdir: Optional[str] = None
    keep_runs: bool = False
    reuse_guess: bool = True
    supports_point_charges: bool = True
    n_parallel: int = 1          # one GPU: runs one at a time

    def __post_init__(self):
        self._cmd = terachem_command(self.command)
        if not self._cmd:
            raise ElectronicStructureError(msg=INSTALL)
        ensure_license_env()
        self._lock = threading.Lock()
        self._guess: dict = {}
        self.last_charges = None
        self.last_stdout = ""

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_lock"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- input
    def tc_input(self, charge: int, multiplicity: int, point_charges: bool, guess: Optional[str]) -> str:
        method = str(self.method).lower()
        if multiplicity != 1 and not method.startswith(("u", "ro")) and method != "hf":
            method = "u" + method       # TeraChem: unrestricted for open shells
        elif multiplicity != 1 and method == "hf":
            method = "uhf"
        lines = ["coordinates geom.xyz", f"method {method}", f"basis {self.basis}", f"charge {int(charge)}",
                 f"spinmult {int(multiplicity)}", "run gradient", "scrdir scr", "units angstrom"]
        if point_charges:
            # The MM charges among themselves are the force field's business.
            lines += ["pointcharges pc.xyz", "pointcharges_self_interaction false"]
        if guess:
            lines.append(f"guess {guess}")
        fixed = {line.split()[0] for line in lines}
        for k, v in dict(self.keywords or {}).items():
            if str(k).lower() not in fixed:
                lines.append(f"{k} {v}")
        return "\n".join(lines) + "\nend\n"

    @staticmethod
    def geometry_xyz(symbols, coords_bohr) -> str:
        x = np.asarray(coords_bohr, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR
        return f"{len(x)}\n\n" + "".join(f"{s} {a:.10f} {b:.10f} {c:.10f}\n" for s, (a, b, c) in zip(symbols, x))

    @staticmethod
    def point_charges_xyz(point_charges) -> str:
        """`charge x y z` per line (Å), after an xyz-style count and comment."""
        rows = [(float(q), np.asarray(xyz, dtype=float) / ANGSTROM_TO_BOHR) for q, xyz in point_charges]
        return f"{len(rows)}\n\n" + "".join(f"{q:.6f} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n" for q, p in rows)

    # --------------------------------------------------------------- run
    def energy_gradient(self, symbols, coords_bohr, charge: int = 0, multiplicity: int = 1,
                        point_charges=None) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
        """(energy Eh, gradient Eh/bohr, gradient on the point charges or
        None). `point_charges`: [(q, (x, y, z) in bohr), ...]."""
        symbols = [str(s) for s in symbols]
        pcs = list(point_charges or [])
        key = (tuple(symbols), int(charge), int(multiplicity))
        with self._lock:
            base = Path(self.workdir) if self.workdir else None
            if base:
                base.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix="tc_", dir=base))
            try:
                guess = None
                if self.reuse_guess and key in self._guess:
                    for name, blob in self._guess[key].items():
                        (run / name).write_bytes(blob)
                    guess = " ".join(sorted(self._guess[key]))      # c0, or ca0 cb0
                (run / "geom.xyz").write_text(self.geometry_xyz(symbols, coords_bohr))
                if pcs:
                    (run / "pc.xyz").write_text(self.point_charges_xyz(pcs))
                (run / "tc.in").write_text(self.tc_input(charge, multiplicity, bool(pcs), guess))
                try:
                    proc = subprocess.run([*self._cmd, "tc.in"], cwd=run, capture_output=True, text=True,
                                          timeout=self.timeout, env=os.environ.copy())
                except subprocess.TimeoutExpired as exc:
                    raise ElectronicStructureError(msg=f"TeraChem did not finish within {self.timeout:.0f} s") from exc
                text = (proc.stdout or "") + "\n" + (proc.stderr or "")
                self.last_stdout = text
                try:
                    energy = parse_energy(text)
                    g_qm, g_pc = parse_gradients(text)
                except ElectronicStructureError:
                    raise ElectronicStructureError(msg=f"TeraChem failed (exit {proc.returncode}):\n{_failure(text)}")
                if len(g_qm) != len(symbols):
                    raise ElectronicStructureError(
                        msg=f"TeraChem gave {len(g_qm)} QM gradient rows for {len(symbols)} atoms")
                if pcs and len(g_pc) != len(pcs):
                    raise ElectronicStructureError(
                        msg=f"TeraChem gave {len(g_pc)} point-charge gradient rows for {len(pcs)} charges")
                scr = run / "scr"
                orbitals = {n: (scr / n).read_bytes() for n in ("c0", "ca0", "cb0") if (scr / n).is_file()}
                if orbitals and ("c0" in orbitals or len(orbitals) == 2):
                    self._guess[key] = {n: b for n, b in orbitals.items() if n == "c0" or "c0" not in orbitals}
                self.last_charges = parse_mulliken(scr / "charge_mull.xls")
                if not pcs:   # a job's viewer can show this SCF's orbitals (mepd.orbitals)
                    from mepd.orbitals import save_run_orbitals, terachem_label

                    for molden in sorted(scr.glob("*.molden"))[:1]:
                        save_run_orbitals(molden, (run / "geom.xyz").read_text(), "terachem",
                                          terachem_label({"method": self.method, "basis": self.basis}))
            finally:
                if not self.keep_runs:
                    shutil.rmtree(run, ignore_errors=True)
        return energy, np.asarray(g_qm, dtype=float), (np.asarray(g_pc, dtype=float) if pcs else None)

    # ------------------------------------------------------------ engine
    def _compute(self, chain: Union[Chain, List]) -> list:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        results = []
        for n in todo:
            e, g, _ = self.energy_gradient(list(n.symbols), n.coords, int(n.structure.charge),
                                           int(n.structure.multiplicity))
            results.append(FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                {"energy": e, "gradient": g})}))
        if todo:
            update_node_cache(node_list=todo, results=results)
        return nodes

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.gradient for n in self._compute(chain)])

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.energy for n in self._compute(chain)])

    def _ase(self, node):
        from mepd.engines.ase import ASEEngine
        from mepd.engines.gxtb import _GXTBASEResultsCalculator

        return ASEEngine(calculator=_GXTBASEResultsCalculator(self, charge=int(node.structure.charge),
                                                              multiplicity=int(node.structure.multiplicity)))

    def compute_geometry_optimization(self, node, keywords: dict | None = None):
        return self._ase(node).compute_geometry_optimization(node, keywords=dict(keywords or {}))

    def compute_transition_state(self, node, keywords: dict | None = None):
        return self._ase(node).compute_transition_state(node=node, keywords=keywords)

    def compute_irc_chain(self, ts_node, keywords: dict | None = None):
        return self._ase(ts_node).compute_irc_chain(ts_node=ts_node, keywords=keywords)
