"""A metal centre's stereo tag from 3D (@SP, @TB, @OH) records which coordination polytope RDKit fitted
to the geometry. For a centre whose ligands are all alike but one (here IrCl4L), no stereoisomers exist,
yet the tag flipped between two geometries of one molecule (@TB6 vs @OH7), so `compare_structures`
called them different: Transition1x-TMC TMC_07727_Ir, its reactant and the g-xTB minimized IRC end of
its re-converged TS. Such tags are now dropped; real cis/trans isomers at a metal stay different."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pynauty")

XYZ = Path(__file__).parent / "data" / "metal_center_fluxional.xyz"


def _frames(path, charge, multiplicity):
    from qcdata import Structure

    from mepd.nodes.node import StructureNode
    from mepd.units import ANGSTROM_TO_BOHR

    lines = path.read_text().splitlines()
    n, out = int(lines[0]), []
    for i in range(0, len(lines), n + 2):
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        out.append(StructureNode(structure=Structure(
            symbols=[r[0] for r in rows], geometry=np.array([[float(v) for v in r[1:4]] for r in rows]) * ANGSTROM_TO_BOHR,
            charge=charge, multiplicity=multiplicity)))
    return out


def test_fluxional_metal_center_is_the_same_molecule():
    import mepd

    reactant, irc_end = _frames(XYZ, 0, 2)
    c = mepd.compare_structures(reactant, irc_end)
    assert c.same, c.reason


def _square_planar(cis: bool):
    """PtCl2Br2(2-), square planar, Pt-X 2.4 A: cis or trans."""
    from qcdata import Structure

    from mepd.nodes.node import StructureNode
    from mepd.units import ANGSTROM_TO_BOHR

    pos = [(2.4, 0, 0), (0, 2.4, 0), (-2.4, 0, 0), (0, -2.4, 0)]
    ligands = ["Cl", "Cl", "Br", "Br"] if cis else ["Cl", "Br", "Cl", "Br"]
    return StructureNode(structure=Structure(symbols=["Pt", *ligands],
                                             geometry=np.array([(0, 0, 0), *pos]) * ANGSTROM_TO_BOHR,
                                             charge=-2, multiplicity=1))


def test_cis_and_trans_at_a_metal_stay_different():
    import mepd

    cis, trans = _square_planar(True), _square_planar(False)
    assert mepd.compare_structures(cis, cis).same
    assert not mepd.compare_structures(cis, trans).same
