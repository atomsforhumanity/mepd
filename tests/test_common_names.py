"""Structures a calculation finds get everyday names ("[OH]" -> hydroxyl),
and a complex's SMILES is its molecules' (H + OH is not read as H+ + OH-)."""
import json

import numpy as np
from qcref.constants import ANGSTROM_TO_BOHR
from qcdata import Structure

from mepd.web import chem
from mepd.web.workspace import Workspace


def _s(symbols, xyz, charge=0, mult=1):
    return Structure(symbols=symbols, geometry=np.asarray(xyz, float) * ANGSTROM_TO_BOHR,
                     charge=charge, multiplicity=mult)


def test_every_name_key_is_a_smiles_and_no_two_spellings_disagree():
    seen = {}
    for smi, name in chem._COMMON_NAMES.items():
        key = chem._canonical(smi)
        assert key is not None, smi
        assert seen.setdefault(key, name) == name, (smi, seen[key], name)


def test_radicals_and_ions_have_their_own_names():
    assert chem.common_name("[OH]") == "hydroxyl"
    assert chem.common_name("[OH-]") == "hydroxide"
    assert chem.common_name("[H][H]") == chem.common_name("[HH]") == "H2"
    assert chem.common_name("OC=O") == "formic acid"        # any spelling
    assert chem.common_name("CCCCCCCCO") is None


def test_a_molecule_named_by_its_smiles_gets_its_common_name(tmp_path):
    ws = Workspace(tmp_path / "ws")
    oh = _s(["O", "H"], [[0, 0, 0], [0.97, 0, 0]], mult=2)
    rec = ws.add_or_merge(oh, name="[OH]", smiles="[OH]", origin={"kind": "job", "job": "j_x"})["rec"]
    assert rec["name"] == "hydroxyl"
    typed = ws.add_or_merge(_s(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
                            name="my water", smiles="O", origin={"kind": "smiles"})["rec"]
    assert typed["name"] == "my water"                       # a name someone typed stays


def test_a_complex_takes_its_molecules_smiles_not_ions(tmp_path):
    ws = Workspace(tmp_path / "ws")
    # H atom 3 A from OH, singlet overall: read as one geometry this can come out "[H+].[OH-]"
    pair = _s(["O", "H", "H"], [[0, 0, 0], [0.97, 0, 0], [4.0, 0, 0]])
    rec = ws.add_or_merge(pair, smiles="[H+].[OH-]", origin={"kind": "job", "job": "j_x"})["rec"]
    assert rec["role"] == "complex"
    members = [ws.structure(m) for m in rec["members"]]
    assert sorted(m["smiles"] for m in members) == ["[H]", "[OH]"]
    assert sorted(rec["smiles"].split(".")) == ["[H]", "[OH]"]
    assert rec["name"] in ("H atom + hydroxyl", "hydroxyl + H atom")


def test_an_older_workspace_is_tidied_on_load(tmp_path):
    ws = Workspace(tmp_path / "ws")
    oh = ws.add_structure(_s(["O", "H"], [[0, 0, 0], [0.97, 0, 0]], mult=2), name="x", smiles="[OH]",
                          merge=False, origin={"kind": "job", "job": "j_x"})
    h = ws.add_structure(_s(["H"], [[0, 0, 0]], mult=2), name="x", smiles="[H]", merge=False,
                         origin={"kind": "job", "job": "j_x"})
    cx = ws.add_complex(_s(["O", "H", "H"], [[0, 0, 0], [0.97, 0, 0], [4.0, 0, 0]]), [oh["id"], h["id"]],
                        origin={"kind": "job", "job": "j_x"})
    fp = tmp_path / "ws" / "workspace.json"
    data = json.loads(fp.read_text())
    data["structures"][oh["id"]]["name"] = "[OH]"            # named by SMILES, as before
    data["structures"][h["id"]]["name"] = "[H]"
    data["structures"][cx["id"]].update(name="[OH] + [H]" if cx["members"][0] == oh["id"] else "[H] + [OH]",
                                        smiles="[H+].[OH-]")
    fp.write_text(json.dumps(data))
    st = Workspace(tmp_path / "ws").snapshot()["structures"]
    assert st[oh["id"]]["name"] == "hydroxyl" and st[h["id"]]["name"] == "H atom"
    assert sorted(st[cx["id"]]["smiles"].split(".")) == ["[H]", "[OH]"]
    assert set(st[cx["id"]]["name"].split(" + ")) == {"hydroxyl", "H atom"}


def test_a_lone_atom_is_not_given_hydrogens():
    assert chem.perceive_smiles(_s(["H"], [[0, 0, 0]], mult=2)) == "[H]"       # not "[HH]" (H2)
    assert chem.perceive_smiles(_s(["Cl"], [[0, 0, 0]], charge=-1)) == "[Cl-]"
    assert chem.common_name(chem.perceive_smiles(_s(["H"], [[0, 0, 0]], mult=2))) == "H atom"
