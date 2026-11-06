"""Run mepd's LangevinMD (reactor trajectories, their quench) on axtb instead of tblite GFN2-xTB.
install(method): "gxtb-v2.0.1" (g-xTB 2.0.1, the program's conventions) or "gfn2" (axtb's GFN2-xTB).
Fermi smearing 3000 K (as mepd's tblite setup); an SCF that fails is retried at 6000 K, then 1000 K
(g-xTB at 3000 K oscillates for some torn geometries, e.g. CO + OH + H, where 6000 K and 0 K converge).
1 thread, Calculators per LangevinMD. `self.scf_retries` counts the retries."""
import numpy as np

LADDER = (3000.0, 6000.0, 1000.0)
RETRIES = [0]                    # all LangevinMDs of this process


def install(method="gxtb-v2.0.1"):
    import axtb
    from rdkit import Chem

    from mepd.discovery import langevin as lg

    if getattr(lg.LangevinMD, "_axtb_method", None) == method:
        return
    init0 = lg.LangevinMD.__init__

    def init(self, symbols, pos_bohr, charge, multiplicity, **kw):
        init0(self, symbols, pos_bohr, charge, multiplicity, **kw)
        table = Chem.GetPeriodicTable()
        uhf = int(multiplicity) - 1
        self._ax_args = ([table.GetAtomicNumber(s) for s in symbols],
                         dict(charge=float(charge), uhf=uhf if uhf else None, threads=1,
                              **({"conventions": "gxtb"} if method.startswith("gxtb") else {})))
        self._ax = {}
        self.scf_retries = 0

    def calc(self, etemp, x):
        c = self._ax.get(etemp)
        if c is None:
            Z, kws = self._ax_args
            c = self._ax[etemp] = axtb.Calculator(method, Z, np.ascontiguousarray(x), etemp=etemp, **kws)
        return c

    def forces(self, x, biased=True):
        from mepd.discovery.nanoreactor import _wall

        x = np.ascontiguousarray(x)
        for k, etemp in enumerate(LADDER):
            try:
                r = calc(self, etemp, x).singlepoint(x, gradient=True)
                break
            except Exception:
                if k == len(LADDER) - 1:
                    raise
                self.scf_retries += 1
                RETRIES[0] += 1
        e, g = float(r["energy"]), np.asarray(r["gradient"], dtype=float).reshape(x.shape)
        if self.wall is not None:
            ew, gw = _wall(x, self.wall / lg.BOHR, self.wall_force)
            e, g = e + ew, g + gw
        if biased and self.bias is not None:
            g = g + self.bias(x)
        return e, g

    lg.LangevinMD.__init__ = init
    lg.LangevinMD.forces = forces
    lg.LangevinMD._axtb_method = method
