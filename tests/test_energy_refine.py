"""Energies of an explored network at a higher level (mepd.discovery.energy_refine)."""
import numpy as np
import pytest
from qcdata import Structure

from mepd.discovery.energy_refine import refine_energies
from mepd.discovery.kinetics import Step

KCAL = 1 / 627.5094740631


def _h2(tag: float) -> Structure:
    """An H2 whose first coordinate tells the fake engines which point it is."""
    return Structure(symbols=["H", "H"], geometry=np.array([[tag, 0.0, 0.0], [tag, 0.0, 1.4]]))


class _Table:
    """A fake engine: energy (Hartree) looked up by the structure's tag."""

    def __init__(self, table: dict):
        self.table = table

    def compute_energies(self, nodes):
        return np.array([self.table[round(float(n.structure.geometry[0][0]), 3)] for n in nodes])


def _network(tmp_path):
    """Seed 0 -> A (1) or B (2), each over its own TS; low-level TS(A) lower."""
    species = [_h2(0.0), _h2(1.0), _h2(2.0)]
    steps = []
    for k, tag in ((1, 10.0), (2, 20.0)):
        fp = tmp_path / f"ts{k}.xyz"
        _h2(tag).save(fp)
        steps.append(Step(0, k, 0.0, f"0->{k}", {"ts": str(fp)}))
    low = {0.0: 0.0, 1.0: -10 * KCAL, 2.0: -10 * KCAL, 10.0: 20 * KCAL, 20.0: 21 * KCAL}
    for st, tag in zip(steps, (10.0, 20.0)):
        st.ts_energy = low[tag]
    return species, steps, low


def _run(species, steps, low, high):
    return refine_energies(species, steps, low_engine=_Table(low), high_engine=_Table(high), temperature=298.15,
                           time_s=3600.0, free_energy=False)


def test_a_constant_shift_between_methods_changes_nothing(tmp_path):
    species, steps, low = _network(tmp_path)
    out = _run(species, steps, low, {k: v - 400.0 for k, v in low.items()})
    for row in out["refined"]:
        assert row["g_high_kcal"] == pytest.approx(row["g_low_kcal"], abs=1e-6)
    assert np.allclose(out["kinetics"]["high"]["final_concentration"],
                       out["kinetics"]["low"]["final_concentration"], atol=1e-9)


def test_the_higher_level_decides_which_product_forms(tmp_path):
    species, steps, low = _network(tmp_path)
    high = {k: v - 400.0 for k, v in low.items()}
    high[20.0] -= 2 * KCAL                  # at the higher level TS(B) is 1 kcal/mol under TS(A)
    out = _run(species, steps, low, high)
    lo, hi = out["kinetics"]["low"]["final_concentration"], out["kinetics"]["high"]["final_concentration"]
    assert lo[1] > lo[2] and hi[2] > hi[1]
    ratio = hi[2] / hi[1]                   # kinetic control: exp(1 kcal/mol / RT)
    assert ratio == pytest.approx(np.exp(1.0 / (0.0019872043 * 298.15)), rel=0.02)
    ts = {r["label"]: r["g_high_kcal"] for r in out["refined"] if r["kind"] == "ts"}
    assert ts["0->2"] == pytest.approx(19.0) and ts["0->1"] == pytest.approx(20.0)


def test_species_out_of_reach_keep_their_energies_on_the_higher_scale(tmp_path):
    species, steps, low = _network(tmp_path)
    steps[1].ts_energy = low[20.0] = 80 * KCAL   # B behind a barrier over the cutoff
    out = refine_energies(species, steps, low_engine=_Table(low),
                          high_engine=_Table({k: v - 400.0 for k, v in low.items()}), temperature=298.15,
                          time_s=3600.0, cutoff_kcal=40.0, free_energy=False)
    assert {r["index"] for r in out["refined"] if r["kind"] == "species"} == {0, 1}
    assert out["kinetics"]["high"]["final_concentration"][2] < 1e-12
    assert out["kinetics"]["high"]["final_concentration"][1] == pytest.approx(
        out["kinetics"]["low"]["final_concentration"][1])


def test_a_charged_complex_counts_as_its_ions(monkeypatch):
    """Acetate next to water (charge -1): two molecules, the charge on acetate."""
    import shutil

    from mepd.discovery.free_energy import FreeEnergies

    if shutil.which("xtb") is None:
        pytest.skip("needs xtb")
    sym = ["C", "C", "O", "O", "H", "H", "H", "O", "H", "H"]
    xyz = np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.15, 1.08, 0.0], [2.15, -1.08, 0.0],
                    [-0.38, 1.02, 0.0], [-0.38, -0.51, 0.89], [-0.38, -0.51, -0.89],
                    [4.9, 1.5, 0.0], [5.4, 2.3, 0.0], [5.4, 0.7, 0.0]])
    s = Structure(symbols=sym, geometry=xyz * 1.8897259886, charge=-1, multiplicity=1)
    parts = FreeEnergies(None, 298.15).parts(s, False)
    assert sorted((len(p), q, m) for p, q, m in parts) == [(3, 0, 1), (7, -1, 1)]


class _Counted(_Table):
    def __init__(self, table):
        super().__init__(table)
        self.calls = 0

    def compute_energies(self, nodes):
        self.calls += len(nodes)
        return super().compute_energies(nodes)


def test_a_rerun_picks_up_what_was_kept_at_the_same_level_only(tmp_path):
    species, steps, low = _network(tmp_path)
    high = {k: v - 400.0 for k, v in low.items()}
    kw = dict(temperature=298.15, time_s=3600.0, free_energy=False, save_dir=tmp_path / "refined")
    first = refine_energies(species, steps, low_engine=_Table(low), high_engine=_Counted(high), **kw)
    again = _Counted(high)
    second = refine_energies(species, steps, low_engine=_Table(low), high_engine=again, **kw)
    assert again.calls == 0
    assert [r["g_high_kcal"] for r in second["refined"]] == pytest.approx([r["g_high_kcal"] for r in first["refined"]])
    other = _Counted({k: v - 1.0 for k, v in high.items()})     # another level: nothing reused
    refine_energies(species, steps, low_engine=_Table(low), high_engine=other, **kw)
    assert other.calls > 0


def test_a_reconvergence_is_kept_as_soon_as_it_ends(tmp_path, monkeypatch):
    from mepd.discovery import energy_refine

    species, steps, low = _network(tmp_path)
    high = {k: v - 400.0 for k, v in low.items()}
    done = []

    def reopt(item, engine, free):
        done.append((item.kind, item.index))
        return {"e_high": high[round(float(item.structure.geometry[0][0]), 3)] - KCAL,
                "g_high": high[round(float(item.structure.geometry[0][0]), 3)] - KCAL, "structures": list(item.parts)}

    monkeypatch.setattr(energy_refine, "_reoptimize", reopt)
    kw = dict(temperature=298.15, time_s=3600.0, free_energy=False, save_dir=tmp_path / "refined", mode="reoptimize")
    first = refine_energies(species, steps, low_engine=_Table(low), high_engine=_Table(high), **kw)
    assert done[0][0] == "ts"                        # the slow ones start first
    assert len(list((tmp_path / "refined" / "reoptimized").glob("*.json"))) == len(done) == 5
    done.clear()
    second = refine_energies(species, steps, low_engine=_Table(low), high_engine=_Table(high), **kw)
    assert done == []
    assert [r["how"] for r in second["refined"]] == ["reoptimized"] * 5
    assert [r["g_high_kcal"] for r in second["refined"]] == pytest.approx([r["g_high_kcal"] for r in first["refined"]])
