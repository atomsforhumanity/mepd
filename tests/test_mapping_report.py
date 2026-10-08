"""The atom-mapping record for the Results page (mepd.mapping_report) and its view."""
from __future__ import annotations

import json

import numpy as np
from qcdata import Structure

from mepd.atom_mapping_selection import MappingCandidate
from mepd.mapping_report import build_report, candidate_drawings, write_report

# Acetaldehyde, and vinyl alcohol in its atom order (C1 C2 O3, H4-6 on C1, H7 on C2).
START = """7

C  -1.2380  0.1470 0.0000
C   0.2200 -0.2410 0.0000
O   1.1300  0.5600 0.0000
H  -1.7900 -0.7900 0.0000
H  -1.4900  0.7400 0.8800
H  -1.4900  0.7400 -0.8800
H   0.4300 -1.3200 0.0000
"""
END = """7

C  -1.2300  0.2000 0.0000
C   0.0800 -0.1000 0.0000
O   1.1000  0.7900 0.0000
H  -1.9900 -0.5800 0.0000
H  -1.5500  1.2400 0.0000
H   1.9500  0.3300 0.0000
H   0.4000 -1.1400 0.0000
"""


def _swap(structure, i, j):
    g = np.asarray(structure.geometry).copy()
    g[[i, j]] = g[[j, i]]
    return structure.model_copy(update={"geometry": g})


def test_report_has_bond_changes_differences_and_frames(tmp_path):
    start, end = Structure.from_xyz(START), Structure.from_xyz(END)
    other = _swap(end, 4, 5)          # the same mechanism with another methyl H moving
    cands = [MappingCandidate(label="identity", end_structure=end, atom_map=None),
             MappingCandidate(label="mapping_0", end_structure=other, atom_map=None)]
    report = build_report(start, end, cands, metric="rmsd", scores={"identity": 1.2, "mapping_0": 0.8},
                          chosen="mapping_0", summary="two compared")
    first, second = report["candidates"]
    assert first["label"] == "mapping_0" and first["chosen"]              # the one used comes first
    assert first["broken"] == [[0, 4]] and first["formed"] == [[2, 4]]    # C1-H5 breaks, O3-H5 forms
    assert second["broken"] == [[0, 5]] and second["formed"] == [[2, 5]]
    assert second["differs_at"] == [4, 5] and first["differs_at"] == []
    assert len(first["frames"]) == 9 and first["frames"][0].startswith("7\n")
    write_report(tmp_path, report)
    assert json.loads((tmp_path / "atom_mapping.json").read_text())["chosen"] == "mapping_0"
    drawings = candidate_drawings(report, first)
    assert drawings["start"] and "<svg" in drawings["start"] and drawings["end"]


def test_results_page_gets_the_mapping_with_drawings(tmp_path):
    from mepd.web.results import atom_mapping_view

    start, end = Structure.from_xyz(START), Structure.from_xyz(END)
    write_report(tmp_path, build_report(start, end, [MappingCandidate("identity", end, None)], metric="",
                                        chosen="identity"))
    view = atom_mapping_view(tmp_path)
    assert view["candidates"][0]["svg"]["end"] and view["symbols"][2] == "O"
    assert atom_mapping_view(tmp_path / "nothing") is None
