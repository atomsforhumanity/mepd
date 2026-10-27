---
name: mepd
description: Drive mepd, AFH's reaction-path tool: mepd run (MSMEP, FSM-TS, NEB, GSM), mepd ts (TS + IRC), mepd channels (conformer/mechanism channel search), networks, nanoreactor, discovery expand, QM/MM in solvent or proteins, and the mepd Python API. Use for any reaction path, minimum energy path, transition state search, barrier, reaction network or mepd profile (.toml) task.
---

# Using mepd

mepd finds reaction paths and transition states: give it two minima (or a SMILES pair) and it
finds the elementary steps between them, each step's TS, and the IRC that proves the TS connects
them. It also explores (conformer channels, networks, nanoreactor) and runs on QM/MM systems.

**Use mepd when** the question is about a reaction path, a TS, a barrier or a network.
**Script the program directly with qc\*** ([`qc-suite`](https://github.com/atomsforhumanity/research-skills/blob/main/plugins/afh-qc/skills/qc-suite/SKILL.md))
when you only need single points, plain optimizations or frequencies at a level mepd has no
engine for. The TS protocol itself (one imaginary mode, IRC must connect, barriers from the lowest
reactant conformer) lives in [`ts-optimization-irc`](https://github.com/atomsforhumanity/research-skills/blob/main/plugins/afh-qc/skills/ts-optimization-irc/SKILL.md);
mepd is one implementation of it. Read [LESSONS.md](LESSONS.md) before a new kind of run.

## Install and check

`uv sync` in a clone (or `pip install mepd`) installs everything, g-xTB included (fetched on first
use). `mepd --help` lists the commands; every command has `--help` with its defaults. `mepd web`
starts the browser UI.

## Commands

| Command | What it does (defaults) |
|---|---|
| `mepd run --start A --end B -i profile.toml -o out` | One path search; `--recursive` (MSMEP: split at intermediates until every leaf is one elementary step), `--use-tsopt --irc` then optimizes each leaf's TS and runs its IRC. |
| `mepd ts --guess X -i profile.toml --irc` | TS optimization (Sella, fmax 0.001 eV/Å, 1000 steps, exactly one imaginary mode < -10 cm⁻¹ required) and IRC (fmax 0.01). `--guess` may be an xyz, a tree dir, a channels dir or a network.json. |
| `mepd channels --start A --end B` | Conformers of both ends (RDKit, or `--backend crest`), every pair searched with MSMEP, TS + IRC per leaf, then classified: `channels/` (one step A→B), `alternate-channels/` (via intermediates), `offtarget-exit-channels/`, `network.json`. |
| `mepd optimize X` | Minimizations, Hessian-checked. |
| `mepd network-build`, `network-splits` | Networks from trees / from a set of minima. |
| `mepd discovery expand / nanoreactor / vri / hessian-sample / refine-energies` | Exploration: graph-rule or nanoreactor products, valley-ridge inflections, higher-level energies of a network. |
| `mepd qmmm build / region / protein-sites / protein-build / reaction / reactor / inspect` | QM/MM systems in solvent or a protein, and reactions in them (see docs/qmmm.md). |
| `mepd retro plan TARGET` | Retrosynthesis with local models. |
| `mepd visualize out` | Interactive view of any output. |

Endpoints are xyz files or SMILES (embedded, then minimized). Both ends must list the same
atoms; mepd checks their atom mapping (`--no-atom-mapping` when they already share atom order,
e.g. pre-mapped datasets).

## Profiles (`-i profile.toml`)

`mepd init -o profile.toml` writes every default. The keys that matter most:

- `engine_name`: `gxtb` (default, g-xTB), `xtb` (GFN2/GFN1, takes point charges), `mlip`,
  `fairchem`, `ase` (any ASE calculator), `psi4`, `orca`, `terachem`, `qccompute`/`chemcloud`;
  options in `[<name>_engine_kwds]`.
- `path_min_method`: `FSM-TS` (default: freezing string, TS + pseudo-IRC check, GSM/NEB
  fallback), `NEB`, `FNEB`, `GSM`, `MLPGI`, …; its settings in `[path_min_inputs]`.
- `[ts_inputs]`: `fmax`, `maxiter`, `check_frequencies`, `imaginary_cutoff_cm`.
- `[solvation]`: implicit solvent, e.g. `solvent = "water"` (g-xTB has none of its own: GFN2
  ALPB is added).
- `[qmmm]` + `qmmm_environment` (`relaxed` | `cage` | `mean_force`): QM/MM. In proteins use
  `cage`.
- `[[restraints]]`: `atoms = [i, j]`, `distance` (Å), `force_constant` (kcal/mol/Å², 500):
  harmonic distance restraints on everything the profile runs.
- `[xtb_engine_kwds] electronic_temperature`: hold one electronic temperature for a whole run.

## Python API

```python
import mepd
eng = mepd.engine("gxtb")                       # or mepd.engine_from_profile("profile.toml")
a, b = mepd.from_smiles("C=CC=C"), mepd.load("product.xyz")
a = mepd.minimize(a, eng)                       # Hessian-checked minimum
path = mepd.interpolate(a, b)
mepd.energies(path.nodes, eng)                  # get_ts_node needs the energies
ts = mepd.find_ts(path.get_ts_node(), eng)      # same defaults as `mepd ts`
res = mepd.irc(ts, eng)                         # ends minimized; mepd.connects(res, a, b)
print(mepd.barrier_kcal(ts, a))
found = mepd.channels(a, b, output="out", profile="profile.toml")
```

Units: Bohr and Hartree inside; functions that take or give other units say so in their names.
docs/python-api.md has the rest.

## Where outputs land

- `run -o out`: `mep_output.xyz` (+ `.energies`), `tree/` (MSMEP tree, `tree.json` says why each
  node split or not), `ts_leaf_<i>.xyz`, `ts_leaf_<i>_irc.xyz`; `*_irc_failed.txt` and
  `*_check_failed.txt` say why an IRC failed or a TS is not the reaction's.
- `channels -o out`: `conformers/`, `pairs/pair_<i>_<j>/`, `channels/`, `stats.json`
  (yield and time per stage), `network.json`.
- `ts -o out`: `<label>.xyz`, `<label>_irc.xyz`.

## Known limits

Open issues: https://github.com/atomsforhumanity/mepd/issues. At the time of writing:
- the service (mepd web as a shared deployment) has no constrained conformer sampling or custom
  profiles ([#15](https://github.com/atomsforhumanity/mepd/issues/15));
- g-xTB's own Hessian is unreliable at its default accuracy (mepd uses gradients or a tighter
  setting);
- TeraChem's own QM/MM (`mm = "terachem"`) takes no restraints.

## Sharing what you learn

A mepd quirk, failure or threshold that cost time goes into LESSONS.md by pull request to
atomsforhumanity/mepd (a human reviews it); a bug or missing feature is an issue there.
