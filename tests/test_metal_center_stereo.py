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


# --- metal-ligand contacts left out (metal_ligand_contacts=False) -----------------------------------
# A ligand that binds the metal through another atom, or comes off it, is another species by default.
# Without metal-ligand contacts only covalent bonds and stereochemistry count, as YARP's organometallic
# extension treats dative contacts as zero-order bonds (Li, Hsu, Savoie, ChemRxiv 2026,
# doi:10.26434/chemrxiv.15006374/v1).

DATA = Path(__file__).parent / "data"


def test_a_coordination_change_alone_is_the_same_species_without_metal_ligand_contacts():
    import mepd

    reactant, irc_end = _frames(DATA / "metal_ligand_contact_only.xyz", 0, 1)    # Pt-N 2.62 vs 1.93 A
    strict = mepd.compare_structures(reactant, irc_end)
    assert not strict.graph and strict.metal_ligand_changed is None
    loose = mepd.compare_structures(reactant, irc_end, metal_ligand_contacts=False)
    assert loose.same, loose.reason
    assert loose.metal_ligand_changed is True
    assert mepd.same_molecule(reactant, irc_end, metal_ligand_contacts=False)
    assert mepd.connects([irc_end, irc_end], reactant, reactant, metal_ligand_contacts=False).both


def test_a_covalent_change_stays_different_without_metal_ligand_contacts():
    import mepd

    product, irc_end = _frames(DATA / "metal_complex_covalent_change.xyz", 0, 1)   # a C-C bond differs
    c = mepd.compare_structures(product, irc_end, metal_ligand_contacts=False)
    assert not c.graph and "covalent" in c.reason


def test_metal_free_structures_are_unaffected():
    import mepd

    cis, trans = _square_planar(True), _square_planar(False)
    for a, b in ((cis, cis), (cis, trans)):
        assert mepd.compare_structures(a, b).same == mepd.compare_structures(a, b, metal_ligand_contacts=True).same
