"""Cost of one energy+gradient call along an MD-like sequence: tblite GFN2 vs axtb GFN2 vs axtb g-xTB 2.0.1."""
import time, sys, numpy as np
from rdkit import Chem
B = 1.8897259886
def read(fp):
    L = open(fp).read().splitlines(); n = int(L[0])
    return [l.split()[0] for l in L[2:2+n]], np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2+n]]) * B
for fp in ["seeds/monomer.xyz", "seeds/cluster4.xyz", "seeds/f0007.xyz"]:
    sy, x = read(fp); Z = [Chem.GetPeriodicTable().GetAtomicNumber(s) for s in sy]
    rng = np.random.default_rng(0); xs = [x + rng.normal(scale=0.05, size=x.shape) for _ in range(50)]
    from tblite.interface import Calculator as TB
    tb = TB("GFN2-xTB", np.array(Z), x); tb.set("verbosity", 0); tb.set("temperature", 3000 * 3.166811563e-6)
    res = None; t = time.perf_counter()
    for p in xs:
        tb.update(p); res = tb.singlepoint(res) if res is not None else tb.singlepoint()
    t_tb = (time.perf_counter() - t) / len(xs)
    import axtb
    out = [f"{fp}: tblite GFN2 {t_tb*1e3:.1f} ms"]
    for meth, kw in [("gfn2", dict(etemp=3000.0)), ("gxtb-v2.0.1", dict(etemp=3000.0, conventions="gxtb")), ("gxtb-v2.0.1", dict(conventions="gxtb"))]:
        try:
            c = axtb.Calculator(meth, Z, x, charge=0.0, threads=1, **kw)
            t = time.perf_counter()
            for p in xs: r = c.singlepoint(np.ascontiguousarray(p), gradient=True)
            out.append(f"axtb {meth} {kw} {(time.perf_counter()-t)/len(xs)*1e3:.1f} ms")
        except Exception as exc:
            out.append(f"axtb {meth} {kw} failed: {exc}")
    print("; ".join(out), flush=True)
