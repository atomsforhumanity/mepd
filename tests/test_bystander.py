"""mepd.bystander on a toy pair potential (no electronic structure): the statuses, the shifts' sign, the
rigid-body modes, the driver over steps."""
import numpy as np
import pytest

from mepd import bystander as B

BOHR = 0.529177210903


class Toy:
    """Point charges on the step's atoms (per frame, by their geometry) and on C, a soft wall between the two,
    harmonic bonds inside C. `mode`: "pair" (that), "flat" (no forces at all), "push" (wall only),
    "split" (C's own bond replaced by a repulsion)."""

    def __init__(self, n_x, q_x, q_c, mode="pair"):
        self.n_x, self.q_x, self.q_c, self.mode = n_x, q_x, np.asarray(q_c, dtype=float), mode

    def _eg(self, node):
        x = np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3)
        g = np.zeros_like(x)
        if self.mode == "flat" or len(x) <= self.n_x:
            return 0.0, g
        e = 0.0
        qx = self.q_x(x[:self.n_x])
        for i in range(self.n_x):
            for j in range(self.n_x, len(x)):
                d = x[j] - x[i]
                r = np.linalg.norm(d)
                qq = 0.0 if self.mode in ("push", "split") else qx[i] * self.q_c[j - self.n_x]
                wall = (3.2 / r) ** (12 if self.mode != "push" else 2)
                e += qq / r + 0.01 * wall
                de = -qq / r ** 2 - 0.01 * (12 if self.mode != "push" else 2) * wall / r
                g[j] += de * d / r
                g[i] -= de * d / r
        for j in range(self.n_x, len(x) - 1):          # C is a chain
            d = x[j + 1] - x[j]
            r = np.linalg.norm(d)
            if self.mode == "split":
                e += 1.0 / r
                de = -1.0 / r ** 2
            else:
                e += 0.5 * (r - 1.8) ** 2
                de = r - 1.8
            g[j + 1] += de * d / r
            g[j] -= de * d / r
        return e, g

    def compute_energies(self, nodes):
        # as mepd's engines: a structure that has an energy keeps it
        for n in nodes:
            if n._cached_energy is None:
                n._cached_energy = float(self._eg(n)[0])
        return np.array([n._cached_energy for n in nodes])

    def compute_gradients(self, nodes):
        return np.array([self._eg(n)[1] for n in nodes])

    def compute_hessian(self, node):
        x0 = np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3)
        n = x0.size
        Hm = np.zeros((n, n))
        for k in range(n):
            for s in (1, -1):
                x = x0.copy().ravel()
                x[k] += s * 1e-3
                Hm[k] += s * self._eg(B._node(node.structure.symbols, x.reshape(-1, 3) * BOHR, 0))[1].ravel() / 2e-3
        return Hm


def step(polar_ts=0.6):
    """A four-atom chain; the TS frame is the one whose middle bond is stretched, and more polar there."""
    sym = ["C", "O", "C", "O"]
    r = np.array([[0, 0, 0], [1.25, 0, 0], [2.6, 0.6, 0], [3.85, 0.6, 0]], dtype=float)
    ts, p = r.copy(), r.copy()
    ts[2:] += [0.5, 0.0, 0.0]
    p[2:] += [1.0, 0.0, 0.0]
    frames = {"symbols": sym, "charge": 0, "multiplicity": 1, "r": r.tolist(), "ts": ts.tolist(), "p": p.tolist()}

    def q_x(x_bohr):
        gap = np.linalg.norm(x_bohr[2] - x_bohr[1]) * BOHR
        q = 0.3 + (polar_ts - 0.3) * np.exp(-((gap - 1.88) / 0.25) ** 2)       # largest at the TS frame
        return np.array([q, -q, q, -q])

    def charges(node):
        x = np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3)
        return q_x(x) if len(x) == 4 else np.array([0.4, -0.4])
    return frames, q_x, charges


def molecule():
    return {"symbols": ["H", "F"], "charge": 0, "multiplicity": 1, "conformers": [[[0, 0, 0], [0.95, 0, 0]]]}


@pytest.mark.parametrize("placer", B.PLACERS)
def test_a_polar_ts_binds_the_molecule_more_and_the_barrier_shift_is_negative(placer):
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4]), placer=placer, poses=4, repeats=2,
                      charges=charges, pool=40)
    assert res["status"] == "ok", res["why"]
    b = res["binding"]
    assert b["ts"]["electronic"] < b["r"]["electronic"] < 0
    assert res["barrier_shift"]["electronic"] == pytest.approx(b["ts"]["electronic"] - b["r"]["electronic"])
    assert res["barrier_shift"]["electronic"] < -0.5
    assert res["reaction_shift"]["electronic"] == pytest.approx(b["p"]["electronic"] - b["r"]["electronic"])
    assert len(res["barrier_shift"]["repeats"]) == 2 and res["barrier_shift"]["spread"] >= 0
    assert b["ts"]["spread"] >= 0
    # the free-energy estimate: binding costs translation and rotation, so it is above the electronic one
    assert b["r"]["free"] > b["r"]["electronic"]
    assert len(b["r"]["rigid_modes_cm"]) == 5         # a diatomic C: five rigid-body modes against the step
    assert res["barrier_shift"]["free"] is not None and res["free_method"]
    assert np.asarray(res["complexes"]["ts"]).shape == (6, 3)
    assert np.allclose(np.asarray(res["complexes"]["ts"])[:4], frames["ts"])      # the step's atoms were held
    assert res["gradient_calls"] > 0 and res["poses"] >= 3 * 2 * 4
    import json

    json.dumps(res)


def test_a_molecule_that_does_not_move_is_a_failure_not_ok():
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4], mode="flat"), placer="seeded", poses=3,
                      repeats=1, charges=charges)
    assert res["status"] == "unmoved" and "moved" in res["why"]
    assert res["barrier_shift"]["electronic"] is None
    assert res["binding"]["ts"]["poses"]["unmoved"] == 3


def test_one_frame_without_a_ts_gives_no_barrier_shift():
    frames, q_x, charges = step()
    res = B.bystander({**frames, "ts": None}, molecule(), Toy(4, q_x, [0.4, -0.4]), placer="seeded", poses=3,
                      repeats=2, charges=charges)
    assert res["status"] == "ok" and "ts" not in res["binding"]
    assert res["barrier_shift"]["electronic"] is None and res["reaction_shift"]["electronic"] is not None


def test_a_molecule_pushed_out_of_contact_is_a_failure():
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4], mode="push"), placer="seeded", poses=3,
                      repeats=1, charges=charges, fmax=0.005, steps=400, carry=0, polish=0)
    assert res["status"] == "failed" and "contact" in res["why"]


def test_a_molecule_that_changes_its_bonds_is_reported_as_reacted():
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4], mode="split"), placer="seeded", poses=3,
                      repeats=1, charges=charges)
    assert res["status"] == "reacted"
    assert res["binding"]["r"]["poses"]["reacted"] == 3


def test_a_proposed_pose_is_one_more_start_beside_the_generic_ones():
    frames, q_x, charges = step()
    near = (np.asarray(frames["ts"])[1] + [[0.0, 2.6, 0.0], [0.0, 3.5, 0.0]]).tolist()
    eng = Toy(4, q_x, [0.4, -0.4])
    res = B.bystander(frames, molecule(), eng, placer="seeded", poses=3, repeats=2, charges=charges,
                      starts=[{"frame": "ts", "coords": near}], carry=0, polish=0)
    assert res["binding"]["ts"]["poses"]["relaxed"] == 3 * 2 + 1
    assert res["binding"]["r"]["poses"]["relaxed"] == 3 * 2
    assert "compute_gradients" not in eng.__dict__     # the call counter is taken off the engine again


def test_sites_are_the_atoms_whose_charge_changes_most_and_the_most_polar():
    q = {"r": np.array([0.0, 0.1, -0.5, 0.0, 0.0]), "ts": np.array([0.3, 0.1, -0.5, 0.0, 0.0]),
         "p": np.array([0.0, 0.1, -0.5, 0.0, -0.2])}
    assert B.sites(q, k=2) == [0, 4, 2]


def test_rigid_modes_of_two_atoms_is_their_stretch():
    k = 0.3                                              # Eh/bohr^2
    Hm = np.zeros((6, 6))
    Hm[0, 0] = Hm[3, 3] = k
    Hm[0, 3] = Hm[3, 0] = -k
    w = B.rigid_modes(Hm, ["H", "F"], [[0, 0, 0], [0.95, 0, 0]], 1)
    from mepd.thermo import _masses

    m = _masses(["H", "F"])
    assert len(w) == 1
    assert w[0] == pytest.approx(5140.4871 * np.sqrt(k * (1 / m[0] + 1 / m[1])), rel=1e-6)


def test_steps_come_back_in_the_order_given_and_as_they_finish():
    frames, q_x, charges = step()
    seen = []
    out = B.bystander_steps({"b": frames, "a": frames}, molecule(), lambda: Toy(4, q_x, [0.4, -0.4]),
                            on_result=lambda k, r: seen.append(k), placer="seeded", poses=2, repeats=1,
                            charges=charges, free_energy=False)
    assert list(out) == ["b", "a"] == seen
    assert out["a"]["status"] == "ok"


def test_the_second_level_is_computed_for_the_step_and_the_molecule_too():
    """A structure keeps the energy of the engine that computed it first: the second level must not reuse it."""
    frames, q_x, charges = step()

    class Shifted(Toy):                       # the same surface, every structure 0.01 Eh per atom lower
        def _eg(self, node):
            e, g = super()._eg(node)
            return e - 0.01 * len(node.structure.symbols), g

    kw = dict(placer="seeded", poses=3, repeats=2, charges=charges)
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4]), final_engine=Shifted(4, q_x, [0.4, -0.4]), **kw)
    for f in ("r", "ts", "p"):
        assert res["final"]["binding"][f]["electronic"] == pytest.approx(res["binding"][f]["electronic"], abs=1e-6)
