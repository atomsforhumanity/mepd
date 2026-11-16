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
        out = np.array([self._eg(n)[0] for n in nodes])
        for n, e in zip(nodes, out):
            try:
                n._cached_energy = float(e)
            except Exception:
                pass
        return out

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
    frames = {"r": B._node(sym, r, 0), "ts": B._node(sym, ts, 0), "p": B._node(sym, p, 0)}

    def q_x(x_bohr):
        gap = np.linalg.norm(x_bohr[2] - x_bohr[1]) * BOHR
        q = 0.3 + (polar_ts - 0.3) * np.exp(-((gap - 1.88) / 0.25) ** 2)       # largest at the TS frame
        return np.array([q, -q, q, -q])

    def charges(node):
        x = np.asarray(node.structure.geometry, dtype=float).reshape(-1, 3)
        return q_x(x) if len(x) == 4 else np.array([0.4, -0.4])
    return frames, q_x, charges


def molecule():
    return B._node(["H", "F"], [[0, 0, 0], [0.95, 0, 0]], 0)


@pytest.mark.parametrize("placer", B.PLACERS)
def test_a_polar_ts_binds_the_molecule_more_and_the_barrier_shift_is_negative(placer):
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4]), placer=placer, poses=4, repeats=2,
                      charges=charges, pool=40)
    assert res["status"] == "ok", res["why"]
    f = res["frames"]
    assert f["ts"]["binding_kcal"] < f["r"]["binding_kcal"] < 0
    assert res["barrier_shift_kcal"] == pytest.approx(f["ts"]["binding_kcal"] - f["r"]["binding_kcal"])
    assert res["barrier_shift_kcal"] < -0.5
    assert len(res["spread"]["barrier_shift_kcal"]["values"]) == 2
    assert res["spread"]["barrier_shift_kcal"]["std"] is not None
    # the free-energy estimate: binding costs translation and rotation, so it is above the electronic one
    assert f["r"]["binding_free_kcal"] > f["r"]["binding_kcal"]
    assert len(f["r"]["rigid_modes_cm"]) == 5         # a diatomic C: five rigid-body modes against the step
    assert "barrier_shift_kcal" in res["free"]
    assert np.asarray(f["ts"]["xyz"]).shape == (6, 3)
    assert np.allclose(np.asarray(f["ts"]["xyz"])[:4], B._xyz(frames["ts"]))      # the step's atoms were held


def test_a_molecule_that_does_not_move_is_a_failure_not_ok():
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4], mode="flat"), placer="seeded", poses=3,
                      repeats=1, charges=charges)
    assert res["status"] == "did_not_move"
    assert "barrier_shift_kcal" not in res
    assert res["frames"]["ts"]["poses"]["unmoved"] == 3


def test_a_molecule_pushed_out_of_contact_is_a_failure():
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4], mode="push"), placer="seeded", poses=3,
                      repeats=1, charges=charges, fmax=0.005, steps=400, carry=0)
    assert res["status"] == "no_contact"


def test_a_molecule_that_changes_its_bonds_is_reported_as_reacted():
    frames, q_x, charges = step()
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.4, -0.4], mode="split"), placer="seeded", poses=3,
                      repeats=1, charges=charges)
    assert res["status"] == "reacted"
    assert res["frames"]["r"]["poses"]["reacted"] == 3


def test_a_proposed_pose_is_one_more_start_beside_the_generic_ones():
    frames, q_x, charges = step()
    far = B._xyz(molecule()) + [0, 40.0, 0]             # out of range of the toy's wall: it will not move
    res = B.bystander(frames, molecule(), Toy(4, q_x, [0.0, 0.0]), placer="seeded", poses=3, repeats=2,
                      charges=charges, proposed={"ts": [far]}, carry=0)
    assert res["frames"]["ts"]["poses"]["relaxed"] == 3 * 2 + 1
    assert res["frames"]["r"]["poses"]["relaxed"] == 3 * 2


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
