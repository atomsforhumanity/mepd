# Experimental targets for the explorer benchmarks

Curated 2026-11-05 by the explorer team's curator (branch `team2/references`). The machine-readable version is `benchmarks/systems.yaml`. This file is the human-readable version, with every quoted passage. It replaces `playgroud/exp_benchmarks/REFERENCES.md` (2026-10-23).

**Rules**
- **Judge against experiment only.** Theory and our own earlier runs are never targets. Theory appears here only to say what a comparison must allow for.
- **CONFIRMED** means the number was read in the primary text (full text, accepted manuscript, or abstract).
- **UNCONFIRMED** means it was seen only as a snippet or a secondary citation. Only CONFIRMED targets are scorable.
- **Error** = RT ln(predicted / nearest bound of the range) at the score temperature, 0 inside the range (BRIEF definition). Each target's range is also given in kcal/mol (RT ln ratio). Script: `data/runs/explorer-team/curator/scripts/targets_kcal.py`.
- **tune** systems may be used to choose settings. **held-out** systems are only ever scored.

## Summary

| # | id | System | Split | Phase, conditions | Target (numerator:denominator) | RT ln (kcal/mol) | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | `hcooh` | HCOOH → CO+H2O vs CO2+H2 | tune | gas, 1500 K, 1 atm Ar, ~1 ms | k1/k2 = 20, **10–40** | 6.86–11.00 | CONFIRMED (Elwardany 2015); fall-off, k∞-vs-1 atm caveat |
| 2 | `acetic_acid` | CH3COOH → CH2CO+H2O vs CH4+CO2 | **held-out** | gas, 1500 K, 1 atm Ar | k4/k3 = 2, **1.1–2.9** | 0.28–3.17 | CONFIRMED (Elwardany 2015); fall-off; mechanism-fitted |
| 3 | `methyl_formate` | CH3OCHO → CH3OH+CO vs all others | tune | gas, 1500 K, 1 atm Ar | **≥ 88:12** (one-sided) | ≥ 5.94 | dominance CONFIRMED; floor PROVISIONAL (Ren 2013 not read) |
| 4 | `f0007` | Cp + methyl acrylate, endo:exo | tune | benzene, 80 °C, 1:1 ~0.9 M, 6.5 h | **77:23 ± 0.5** (3.26–3.44) | 0.83–0.87 | CONFIRMED (Lording 2020 + ESI) |
| 5 | `cp_mbl` | Cp + α-methylene-γ-butyrolactone, exo:endo | **held-out** | benzene, 80 °C, 1:1 ~0.9 M, 4 h | **88:12 ± 0.5** (7.0–7.7) | 1.37–1.43 | CONFIRMED (Lording 2020 + ESI) |
| 6 | `isoprene_ma` | isoprene + methyl acrylate, para:meta | tune | toluene, 145 °C, 15 h | 71:29, **68:32–75:25** | 0.63–0.91 | CONFIRMED (Renslo thesis 1998; Treu 2022) |
| 7 | `furan_maleimide` | furan + maleimide, k_endo/k_exo; exo wins at long times | tune | solution (solvent unknown), 300 K | **1.40 ± 0.08** + exo dominant at long times | 0.17–0.23 | values CONFIRMED (abstract); conditions UNCONFIRMED → PROVISIONAL |
| 8 | `cb_diester` | cis-cyclobutene diester → (2Z,4E) vs (2E,4E) diene | **held-out** | neat, 90 °C, 2 h | **≥ 12.2** (one-sided) | ≥ 1.81 | CONFIRMED value; one-sided by choice (see §8) |

Chemistry covered:
- **Unimolecular eliminations:** 1, 2, 3.
- **Bimolecular pericyclic:** 4 and 5 (stereoselectivity) and 6 (regioselectivity).
- **Closed-shell multistep, reversible:** 7.
- **Rearrangement:** 8 (electrocyclic ring opening).

Radical channels appear only in 3 (CH3 + HCO2, flagged). All systems have 5–25 atoms.

## 1. Formic acid: dehydration vs decarboxylation (`hcooh`, tune)

HCOOH → CO + H2O (k1) and HCOOH → CO2 + H2 (k2); gas, Ar bath.

**Target: k1/k2 = 20, window 10–40, at 1500 K, 1 atm Ar, ~0.5 % HCOOH, ~1 ms.**
- In kcal/mol at 1500 K (RT = 2.981): RT ln 10 = 6.86, RT ln 20 = 8.93, RT ln 40 = 11.00. Window width 4.13 kcal/mol.
- The old 6.3–10 window was 1.38 kcal/mol wide and has no source.

### The source of the target
Elwardany, Nasir, Es-sebbar, Farooq, *Proc. Combust. Inst.* 35, 429–436 (2015), [doi:10.1016/j.proci.2014.06.141](https://doi.org/10.1016/j.proci.2014.06.141).
Read in full: the accepted manuscript ("revised v7"), from the KAUST repository copy archived by the Wayback Machine on 2018-07-23. Exponents were checked against the page images.
- Method: simultaneous laser absorption of CO (4.56 µm), CO2 (4.18 µm) and H2O (2.93 µm) behind reflected shocks. It is the only formic acid study that measures CO2 by absorption. Saito 2005 monitored CO2 by IR emission, as Elwardany's introduction says: *"by monitoring the IR emission of formic acid monomer (5.66 µm), CO (4.63 µm) and CO2 (4.23 µm)"*. Hsu monitored CO only.
- Abstract: *"These two decomposition channels were found to be in the fall-off region and have a branching ratio, k1/k2, of approximately 20 over the range of pressures studied here."*
- §3.1: *"Comparing Fig. 4(a) and Fig. 4(b), it can be concluded that the k1/k2 ≅ 20 over the entire temperature range studied here."*
- §3: *"For formic acid, the experiments were performed at temperatures of 1234 - 1821 K, pressures of 1.0 and 6.5 atm and initial concentration of 0.5% in argon."*
- Abstract, first-order rate constants:
  - k1(1 atm) = 1.03×10¹¹ exp(−25651/T) s⁻¹ (±37 %)
  - k1(6.5 atm) = 9.12×10¹² exp(−30275/T) s⁻¹ (±32 %)
  - k2(1 atm) = 1.79×10⁸ exp(−21133/T) s⁻¹ (±41 %)
  - k2(6.5 atm) = 2.73×10⁸ exp(−20074/T) s⁻¹ (±37 %)
- §3.3: *"The overall uncertainty is calculated by the root-sum-squared method and found to be ± 37% and ± 41% for the first-order rate constants of R1 and R2, respectively at 1396 K and 1.22 atm."*

**How 10–40 follows from the paper:**
- The stated per-rate uncertainties give ±55 % on the ratio, i.e. 9–31.
- Their own Arrhenius fits give, at 1500 K, 28 (1 atm) and 37 (6.5 atm); at 1300 K, 18 and 13 (`scripts/formic_ratios.py`).
- A factor of 2 around their stated 20 covers both.

### The other candidates, reconciled
| Candidate | What the primary source says | Verdict |
|---|---|---|
| "6.3–10" (team's earlier window) | No source found in any paper checked. | Dropped. |
| Hsu, Shaub, Blackburn, Lin, *Symp. (Int.) Combust.* 19, 89 (1982), [doi:10.1016/S0082-0784(82)80181-1](https://doi.org/10.1016/S0082-0784(82)80181-1); "~10 at low T → ~1 near 2000 K" | Abstract (NYCU research portal): CO laser only. *"apparent second order rate constants in the pressure range 0.75-2.8 atm. (k_1 = 2.3 x 10^15 exp (- 50, 000 ± 1700 / RT) c c mole^- 1 sec^- 1; k_2 = 1.5 x 10^16 exp (- 57, 000 ± 2800 / RT) c c mole^- 1 sec^- 1)"* (as indexed; the abstract's channel 2 reads "CO_2 + H_2 O", evidently a typo for CO2 + H2). These give k1/k2 = 2.3 at 1300 K, **1.6 at 1500 K** and 0.9 at 2000 K. The "~10" exists only secondhand. Chang, Chen, Xu, Lin, *J. Phys. Chem. A* 111, 6789 (2007), [doi:10.1021/jp069036p](https://doi.org/10.1021/jp069036p) (with M.C. Lin, a Hsu coauthor): *"the ratio of 10 measured experimentally by Hsu et al."*. Elwardany's introduction: *"~10 at low temperatures and approaches 1 at 2000 K"*. | "~10" UNCONFIRMED: not in the abstract, and its expressions contradict it. k2 is inferred from CO profiles only. Not used. |
| Elwardany et al. 2015, "~20" | Quoted above. | **The target.** |
| Saito, Kakumoto, Kuroda, Torii, Imamura, *J. Chem. Phys.* 80, 4989 (1984), [doi:10.1063/1.446521](https://doi.org/10.1063/1.446521); "~100" | Abstract (OpenAlex): channel 2 *"was very small"*; *"On the basis of a RRKM weak collision, k2,0 values were estimated which was smaller than k1,0 by about two orders, being consistent with the experimental results."* | The ~100 is an RRKM estimate, not a measurement. Qualitative support only. |
| Saito, Shiose, Takahashi, Hidaka, Aiba, Tabayashi, *J. Phys. Chem. A* 109, 5352 (2005), [doi:10.1021/jp045072h](https://doi.org/10.1021/jp045072h) | Abstract (PubMed 16839059): dehydration main, decarboxylation minor; *"the dehydration process is in the second-order region and that the decarboxylation is in the falloff region, in the temperature range of 1300-2000 K and over the total density of (0.5-2.5) x 10(-5) mol cm(-3)"* (≈0.6–3 atm at 1500 K). The measured k2/k1 values are in the paywalled full text. | Regime CONFIRMED; numbers UNCONFIRMED. A same-group theory paper (Takahashi et al., *J. Mol. Struct. THEOCHEM* 545, 197, 2001) reportedly says "about forty times", seen only in a search snippet (repository under maintenance). UNCONFIRMED; it would sit at the window's upper edge. |

### Fall-off, and what a high-pressure-limit model must be compared with
**The experiments are in fall-off.** Every study runs at 0.75–6.5 atm in Ar and reports pressure-dependent rate constants:
- Hsu: apparent second-order constants.
- Saito 1984: second-order k1,0.
- Saito 2005: k1 second order, k2 in fall-off.
- Elwardany: both channels in fall-off; from 1 to 6.5 atm at 1500 K, k1 rises ×4 and k2 ×3.

No experiment reaches the high-pressure limit. The ratio itself moves little: Elwardany report ≈20 at both 1 and 6.5 atm. But the studies disagree on which channel is nearer the low-pressure limit, so even the sign of the correction to the ratio is unknown.

**What the explorer must be compared with.** The explorer uses high-pressure-limit TST (k∞), so it has to be compared with the measured 1 atm fall-off ratio, 10–40 at 1500 K, and scored as k∞ (model) vs fall-off (experiment).
- Nobody has measured k1∞/k2∞, so there is no experimental high-pressure target.
- Errors inside about ±1 kcal/mol can't be pinned on the explorer.
- How large the correction could be, from theory (not a target): Chang 2007's RRKM gives k1∞/k2∞ ≈ 14 at 1500 K but k1,0/k2,0 ≈ 1.3.
- A pressure-dependent (master-equation) kinetics option would remove the caveat.

### Other caveats
- Secondary chemistry is minor at 0.5 % HCOOH. Above 1500 K, Elwardany use only the first 50–100 µs.
- Barriers must be measured from the lowest reactant conformer (E/Z formic acid).
- For two parallel first-order channels, CO:CO2 equals k1/k2 at any time.
- **Monomer only.** The window does not apply to the (HCOOH)4 cluster benchmark (`hcooh4`), which has no experimental target. Elwardany report that pre-shock dimers are *"less than 10% in all considered cases"* and that *"The shock-heating process will convert all dimer to monomer at the reflected shock wave (time zero)."*

## 2. Acetic acid: dehydration vs decarboxylation (`acetic_acid`, held-out)

CH3COOH → CH2CO + H2O (k4) and CH3COOH → CH4 + CO2 (k3); gas, Ar bath.

**Target: k4/k3 = 2, window 1.1–2.9, at 1500 K, 1 atm Ar.** In kcal/mol at 1500 K: RT ln 2 = 2.07; window 0.28–3.17.

Source: Elwardany et al. 2015, the same manuscript as formic acid, read in full.
- Abstract: *"The branching ratio, k4/k3, was found to be approximately 2. The decarboxylation and dehydration reactions of acetic acid appear to be in the falloff region over the tested pressure range"*.
- Abstract, first-order rate constants:
  - k3(1 atm) = 3.18×10¹¹ exp(−28679/T) s⁻¹ (±30 %)
  - k3(6 atm) = 3.51×10¹² exp(−31330/T) s⁻¹ (±26 %)
  - k4(1 atm) = 7.9×10¹¹ exp(−29056/T) s⁻¹ (±34 %)
  - k4(6 atm) = 6.34×10¹² exp(−31330/T) s⁻¹ (±31 %)
- These give k4/k3 = 1.9 (1 atm) and 1.8 (6 atm) at 1500 K. The window is 2 × (1 ± 0.45).
- §3.2: *"Our measured rate constant data show that the tendency of the dehydration reaction to occur is about two times higher than that of the decarboxylation reaction, as observed previously by [19-21]. Mackie and Doolan [18], on the other hand, argued that both reactions have about the same probability."*
- Method (§3.2): *"The rate constants, k3 and k4, of these two reactions were determined by fitting the measured CO2 and H2O profiles with simulations from the acetic acid mechanism of Mackie and Doolan [18]."* So the values depend on that mechanism.
- Older values (Mackie & Doolan 1984, about equal; Butkovskaya et al. 1995, 2) are known only through Elwardany's citations: UNCONFIRMED.

Fall-off applies as for formic acid. It is held out because it is formic acid's sibling: it tests whether settings tuned on formic acid transfer.

## 3. Methyl formate (`methyl_formate`, tune, target provisional)

CH3OCHO → CH3OH + CO (R1), 2 CH2O (R2), CH4 + CO2 (R3; also via the carbene, R7), CH2O + H2 + CO (R5); radical channel CH3 + HCO2 (R4).

**Target: one-sided, CH3OH+CO : all other channels ≥ 88:12 (7.33), at 1500 K, 1 atm Ar.**
- RT ln 7.33 = 5.94 kcal/mol at 1500 K.
- PROVISIONAL: the floor rests on Ren 2013's numbers as quoted by Cho 2024. Ren itself was not read.
- The phase-A reference (CH3OH+CO 94.7 %, CH2O+H2+CO 2.2 %, CH3+HCO2 2.1 %, CH4+CO2 0.5 % at 1400 K) is **Cho 2024's theory, not experiment**, and is not a target.

Source: Cho, Labbe, Harding, Klippenstein, Sivaramakrishnan, *Proc. Combust. Inst.* 40, 105684 (2024), [doi:10.1016/j.proci.2024.105684](https://doi.org/10.1016/j.proci.2024.105684). Read in full (OSTI 2566363 accepted manuscript).
- *"Thermal decomposition studies in the literature using reactors and shock tubes [6-8] point to the dominance of the molecular channel producing CH3OH and CO."*
- *"Experimental studies [8] suggest that these reactions are minor channels with branching ratios that vary between 2 – 6 % for R2 and 6 – 4 % for R3 over the 1000 – 2000 K T-range at P ∼ 1 atm."* ([8] = Ren 2013)
- *"Similarly, the CO profiles from the experiments of Ren et al. (Fig. 4C) shows that they exclusively sensitize R1."*
- *"The present simulations conclusively indicate that the CH2O and CH4 profiles measured by Ren et al. [8] are influenced by secondary chemistry and cannot be used to isolate the kinetics for direct molecular channels from MF to these products"*
- Theory, a caveat only: *"In the high-pressure limit, the radical channel R4 begins to compete strongly with the molecular channel R1 with branching ratios exceeding 10 % at T > 1300 K."* and *"at P = 1 atm fall-off de-emphasizes the role of the radical channels and branching ratios to R4 exceed 1 % only at T > 1400 K."*

Not read:
- Ren, Lam, Pyun, Farooq, Davidson, Hanson, *Proc. Combust. Inst.* 34, 453 (2013), [doi:10.1016/j.proci.2012.05.071](https://doi.org/10.1016/j.proci.2012.05.071). Paywalled; the KAUST copy is behind a bot wall. Its k1 expression in the old REFERENCES.md is UNCONFIRMED.
- Peukert, Sivaramakrishnan, Su, Michael, *Combust. Flame* 159, 2312 (2012), [doi:10.1016/j.combustflame.2012.03.007](https://doi.org/10.1016/j.combustflame.2012.03.007), the direct H-ARAS k4.

Abstract only:
- Wenz, Pazdera, Golka, Olzmann, *J. Phys. Chem. A* 127, 1036 (2023), [doi:10.1021/acs.jpca.2c07053](https://doi.org/10.1021/acs.jpca.2c07053): *"temperatures between 1430 and 2070 K at a nominal pressure of 1.1 bar"*, H-ARAS and TOF-MS. Its branching is in the full text.
- Döntgen, Wildenberg, Heufer, *Int. J. Chem. Kinet.* 58, 199 (2026), [doi:10.1002/kin.70035](https://doi.org/10.1002/kin.70035). CC-BY, but automated download was refused. It has new laser-absorption data on methyl formate and says formic acid decomposition *"might be very sensitive to the bath gas"*. **Worth downloading by hand.**

Fair comparison:
- Fall-off decides the radical share, so a k∞ model will overstate CH3 + HCO2.
- R4 is a radical channel (open-shell fragments): flag it.
- CH2O and CH4 + CO2 were never measured free of secondary chemistry: count them for recall only.
- CH3OH + CO must be found and connected.

## 4. Cyclopentadiene + methyl acrylate, endo:exo (`f0007`, tune)

**Target: endo:exo = 77:23 ± 0.5 % (3.26–3.44), benzene, 1:1 about 0.9 M each, 80 °C, 6.5 h.** RT ln = 0.848 kcal/mol (range 0.828–0.868) at 353.15 K.

Source: Lording, Fallon, Sherburn, Paddon-Row, *Chem. Sci.* 11, 11915–11926 (2020), [doi:10.1039/d0sc04553e](https://doi.org/10.1039/d0sc04553e). Read in full (PMC8162770).
- Table 1, methyl acrylate 7 + CPD 2: temperature 80 °C, time 6.5 h, isolated yield 86 %, **endo:exo (exp.) 77 : 23**, endo:exo (CBS-QB3, benzene) 76 : 24.
- *"Thus, a 1 : 1 molar ratio of CPD and dienophile (1 M concentration solutions of both diene and dienophile in benzene) were heated at 80 °C in the majority of cases, ... and the DA reactions of the three least reactive dienophiles employing 3 molar equivalents of CPD."* The three are not named in the main text. The ESI shows methyl acrylate run 1:1 (below); it uses 3 equivalents of Cp for acrylonitrile and for the preparative acrylamide run.
- *"Experimental endo : exo ratios were determined by quantitative 800 MHz 1H NMR spectroscopy."*
- ESI, general methods: *"Quoted endo:exo ratios refer to values obtained from 800 MHz 1H NMR spectroscopic analysis of crude reaction mixtures and were recorded in duplicate (maximum difference between runs = +/- 0.5%)."* This is the ±0.5 % used for the range. No systematic error is stated.
- ESI procedure: *"Methyl acrylate (7) (0.18 mL, 1.0 mmol) was added to an ampule followed a solution of cyclopentadiene (2) in benzene (2.0 mL of a 1.0M solution in PhH, 2.0 mmol). The ampule was sealed and placed in a heating bath at 80 °C for 6.5 h."* ... *"(total yield 261 mg, 86%)"*. 0.18 mL of methyl acrylate is 2.0 mmol, and 261 mg is 86 % of 2.0 mmol of adduct, so "1.0 mmol" is a typo. The run is 1:1 at about 0.92 M each. The endo:exo ratio was recorded *"from equivalent runs conducted in benzene-d6"*.
- *"For representative DA reactions involving BD and CPD, different endo : exo ratios of selected products were exposed to the reaction conditions under which they were formed. An unchanged ratio was returned in each case, confirming the kinetic control of these reactions."*

Solvent dependence:
- Cativiela, García, Mayoral, Salvatella, *J. Chem. Soc. Perkin Trans. 2* 1994, 847, [doi:10.1039/p29940000847](https://doi.org/10.1039/p29940000847). Abstract only; the full text is restricted. They measured Cp/isoprene + methyl acrylate/MVK in 18 solvents: *"The endo/exo selectivity mainly depends on the solvophobic (Sp) and hydrogen bond donor (α) properties of the solvent. The influence of the dipolarity (π\*) is only observed in the reactions with methyl acrylate."*
- Berson, Hamlet, Mueller, *J. Am. Chem. Soc.* 84, 297 (1962), [doi:10.1021/ja00861a033](https://doi.org/10.1021/ja00861a033), the Ω solvent scale built on this reaction. Not read.
- A secondary compilation (ACS Omega 2021, PMC8340089, Table 6) quotes 2.9:1 in Et2O and 5.2:1 in MeOH at 30 °C. UNCONFIRMED.

Fair comparison:
- The solvent is benzene; a gas-phase model is compared with a solution value. Lording's CBS-QB3 shows little gas vs benzene-PCM change for methyl acrylate (theory).
- Kinetic control is confirmed, so the ratio of formation rates is what counts.
- Count both enantiomers of each adduct (mirror-image TSs) and both dienophile conformers (s-cis/s-trans), i.e. four TSs.
- Cp dimerization competes for Cp; it changes conversion, not endo:exo.
- The 0.85 kcal/mol selectivity is smaller than the TS-conformer noise seen in phase A (1.5 kcal/mol between mirror-image TSs).

## 5. Cyclopentadiene + α-methylene-γ-butyrolactone, endo:exo (`cp_mbl`, held-out)

**Target: endo:exo = 12:88 ± 0.5 %**, scored as exo:endo = 7.33 (range 7.0–7.7). Benzene, 1:1 (1.0 mmol each, about 0.92 M), 80 °C, 4 h, 72 % yield. RT ln 7.33 = 1.398 kcal/mol at 353.15 K (range 1.366–1.432).

Source: Lording et al. 2020, Table 1, α-methylene γ-butyrolactone 11 + CPD 2: *"2 | 11CPD-n, 11CPD-x | 80 | 4 h | 72 | 12 : 88 | 10 : 90 | 5.3 : 0.0 | 4.5"* (diene | adducts | °C | time | yield % | endo:exo exp. | calc. | ...). Endo/exo refers to the lactone carbonyl. The product is a spiro compound.

It is held out because it is the same reaction class and protocol as F0007, with the opposite selectivity: a method tuned toward endo on F0007 fails here.

## 6. Isoprene + methyl acrylate, para:meta (`isoprene_ma`, tune)

Uncatalysed Diels-Alder.
- para = methyl 4-methylcyclohex-3-ene-1-carboxylate, `COC(=O)C1CC=C(C)CC1`.
- meta = methyl 3-methylcyclohex-3-ene-1-carboxylate, `COC(=O)C1CCC=C(C)C1`.

**Target: para:meta = 71:29 (2.45), window 68:32–75:25 (2.13–3.0), toluene, 145 °C (418.15 K), 15 h.** RT ln = 0.744 kcal/mol (window 0.626–0.913).

Source: A.R. Renslo, PhD thesis, MIT (1998), hdl:1721.1/47485, Part II, Table 6 ("Regioselectivity of Diels-Alder Reactions in Carbon Dioxide and Conventional Solvents"). Read via the DSpace@MIT text; page 148 was checked as an image. 270 is drawn as the 1,4 adduct and 271 as the 1,3 adduct. Ratios are 270:271 by ¹H NMR, with GC in brackets:

| Entry | Conditions | Yield (%) | 270:271 NMR (GC) |
|---|---|---|---|
| 1 | PhCH3, 145 °C, 15 h | 78 | **71:29 (71:29)** |
| 2 | PhCH3, 50 °C, 3 d | (7) | 69:31 (72:28) |
| 3 | CO2 49.5 bar, 50 °C, 4 d | (11) | 69:31 (73:27) |
| 4 | CO2 74.5 bar, 50 °C, 4 d | (5) | 67:33 (73:27) |
| 5 | CO2 95.2 bar, 50 °C, 7 d | (4) | 71:29 (73:27) |
| 6 | CO2 117 bar, 50 °C, 3 d | (3) | 70:30 (72:28) |

Footnotes: *"a Isolated yield or (estimated by 1H NMR). b Ratio of isomers determined by 1H NMR and (GC) analysis."* The text: *"...cycloadducts 270 and 271 in a ratio very similar to that obtained in toluene (entries 1 and 2); i.e., no deviation from normal Diels-Alder regioselectivity had occurred."*

The peer-reviewed version is Renslo, Weinstein, Tester, Danheiser, *J. Org. Chem.* 62, 4530 (1997), [doi:10.1021/jo9704523](https://doi.org/10.1021/jo9704523). Not read; that its numbers match the thesis is UNCONFIRMED.

**Other primary values:**
- Treu, Huber, Pleßow, Studt, Saraçi, *Catal. Sci. Technol.* 12, 7439 (2022), [doi:10.1039/d2cy01337a](https://doi.org/10.1039/d2cy01337a), CC BY. Read in full. *"the para : meta-ratio for the uncatalyzed reaction is 3"*. Cyclohexane, 70 °C, 5 h, isoprene:MA 3:1, about 6 % yield, GC-FID. This sets the window's upper edge.
- US 9,061,989 B1, Comparative Example 2 (after Nazarov et al. 1959). Read in full. Neat with hydroquinone, 120 °C, 6 h: *"74% yield of the crude product and the 79% selectivity of the para product"*. The analysis method is not stated for this example. It lies above the window and is not used.

**Catalysed, not a target:** AlCl3 gives about 95:5. This is Inukai & Kojima, *J. Org. Chem.* 31, 1121 (1966), [doi:10.1021/jo01342a031](https://doi.org/10.1021/jo01342a031), not read. The 95:5 is cited by Vermeeren et al. 2020 (PMC7187354), and the patent's Comparative Example 1 reproduces it (benzene, 10–20 °C, 95 % para).

**Leads:**
- Hennis, *J. Org. Chem.* 28, 2570 (1963), [doi:10.1021/jo01045a020](https://doi.org/10.1021/jo01045a020): "The Independence of Reaction Temperature and Mode of Addition in the Isoprene–Methyl Acrylate Diels-Alder Reaction". Title only.
- Cativiela et al. 1994 (abstract): *"Regioselectivity almost exclusively depends on the hydrogen bond donor ability of the solvent."*

**Fair comparison:**
- Toluene has no H-bond donors, so a gas-phase model is fair here.
- Each regioisomer has one stereocentre and forms as a racemate. All TSs of one regiochemistry (endo/exo, dienophile s-cis/s-trans, both faces) give the same product: sum them.
- 0.74 kcal/mol selectivity is as demanding as F0007.

## 7. Furan + maleimide: kinetic endo, thermodynamic exo (`furan_maleimide`, tune, target provisional)

This is the closed-shell multistep case. The endo adduct forms about as fast as exo, reverts by retro-Diels-Alder, and exo accumulates. Both adducts are meso: endo `O=C1NC(=O)[C@H]2[C@@H]1[C@H]1C=C[C@@H]2O1`, exo `O=C1NC(=O)[C@H]2[C@@H]1[C@@H]1C=C[C@H]2O1`.

**Target (provisional):**
- Initial formation k_endo/k_exo = 1.40 ± 0.08 at 300 K. RT ln = 0.200 kcal/mol (range 0.165–0.234).
- Qualitative: exo is the dominant product at long times.
- Provisional because the solvent, the concentrations and the kinetic scheme behind the rate constants are unknown.

Source: Rulíšek, Šebek, Havlas, Hrabal, Čapek, Svatoš, *J. Org. Chem.* 70, 6295 (2005), [doi:10.1021/jo050759z](https://doi.org/10.1021/jo050759z). Abstract only (PubMed 16050690). The full text and SI are on MPG.PuRe, but restricted to the MPG network.
- *"The experimental values of rate constants (k(1+2endo) = (1.75 +/- 0.48) x 10(-5); mol(-1) l s(-1); k(1+2exo) = (3.10 +/- 0.55) x 10(-5); mol(-1) l s(-1); k(1+3endo) = (1.93 +/- 0.082) x 10(-5); mol(-1) l s(-1), k(1+3exo) = (1.38 +/- 0.055) x 10(-5); mol(-1) l s(-1) all at 300 K)"*. Here 1 = furan, 2 = maleic anhydride, 3 = maleimide.
- *"The low thermodynamic stability of furan [4 + 2] adducts opens retro-Diels-Alder reaction channels and overrules the very small kinetic preference (calculated and measured here) of initial formation for endo stereoisomers. On a macroscopic scale "an irregular"-thermodynamically more stable-exo stereoisomer was consequently observed as a dominant species."*

**Why maleimide and not maleic anhydride.** Maleimide has the tighter rate constants (±4–6 % against ±18–27 %). The literature on furan + maleic anhydride also conflicts. Lennox, …, Day, Friščić, *Chem. Sci.* 17, 15432 (2026), [doi:10.1039/d5sc06724c](https://doi.org/10.1039/d5sc06724c), read in full, summarises it:
- *"Lee and Herndon have indicated that the initial rate of formation for endo-1 is ca. 500 times higher compared to that of exo-1, ... Koreshkov et al. have found that the rates of formation of endo- and exo-1 are comparable, ... More recently, the Svatoš group has observed comparable initial rates of formation for both isomers, with endo-1 subsequently converting into the exo-form."* The cited values are secondary here.
- Their own neat (solvent-free, solidifying) reaction gave endo:exo *"ca. 2 : 7"* after standing overnight at room temperature. That is not a solution benchmark.
- Widstrom & Lear, *Sci. Rep.* (2019), [doi:10.1038/s41598-019-54156-4](https://doi.org/10.1038/s41598-019-54156-4), measured retro-Diels-Alder rates of these adducts by NMR. The rate-constant units are not stated and the isomer was not assigned experimentally (per the source agent), so it is not used.

**Fair comparison:**
- The kinetics must include the retro-Diels-Alder of both adducts. The initial ratio tests the forward TSs; the long-time exo dominance tests the thermodynamics and the reverse steps.
- Don't promote it to scorable until the conditions are read. Needed: the Rulíšek full text, by MPG-network access, an ACS subscription, or a request to the authors.

## 8. Electrocyclic ring opening of dimethyl cis-cyclobut-3-ene-1,2-dicarboxylate (`cb_diester`, held-out)

Reactant: `COC(=O)[C@H]1C=C[C@H]1C(=O)OC` (cis, meso). Products:
- conrotatory (2Z,4E)-muconate, `COC(=O)/C=C\C=C\C(=O)OC`;
- (2E,4E)-muconate, `COC(=O)/C=C/C=C/C(=O)OC`.

**Target: (Z,E):(E,E) ≥ 12.2 (one-sided), neat, 90 °C (363.15 K), 2 h.** RT ln 12.2 = 1.81 kcal/mol.

Source: Sau, Nagarajan, Patrahau, Lethuillier-Karl, Vergauwe, Thomas, Moran, Genet, Ebbesen, *Angew. Chem. Int. Ed.* 60, 5712 (2021), [doi:10.1002/anie.202013465](https://doi.org/10.1002/anie.202013465), PMC7986062. Full text read; quotes re-checked by the curator; SI not read. The paper is about vibrational strong coupling; only its ordinary "non-cavity" control is used.
- *"Neat CB was injected into the preheated chamber at 90 °C and the reaction was monitored in the FT‐IR"* ... *"After two hours, the reaction mixture was allowed to cool to room temperature (RT) and dissolved in CDCl3 to perform the 1H‐NMR measurement."*
- *"As predicted by the WH rules, the symmetry‐allowed cis‐trans butadiene is obtained as the major product (92 %) while the symmetry‐forbidden trans‐trans derivative is present only with a 8 % yield. This leads to a ratio between symmetry‐allowed conrotatory (cis‐trans) vs. symmetry‐forbidden disrotatory (trans‐trans) product ratio of 12.7±0.5. No other products or intermediates were observed in the NMR spectra."*
- Table 1: *"Non‐cavity | 12.7±0.5 | 7.7±0.2"* (cis-trans/trans-trans | ΔΔG‡ in kJ/mol).
- *"In non‐cavity conditions, the average free energy change between con‐ and disrotatory transition states of CB at 90 °C is 7.6±0.2 kJ mol−1, which is much lower when compared to cis‐3,4‐dimethylcyclobutene (35–40 kJ mol−1)."*

**Why one-sided.** A disrotatory TS only 7.6 kJ/mol (1.8 kcal/mol) above the allowed one is an extraordinary claim; the authors' own comparison case has a 35–40 kJ/mol gap. The text read does not exclude other sources of the 8 % (E,E): a trans-cyclobutene impurity (its allowed opening gives E,E), or surface chemistry in the 6 µm film between SiOx windows. A two-sided target of 12.7 would punish a correct stereospecific prediction if that 8 % is an artefact. So only the lower bound is scored: the conrotatory product must dominate by at least 12.2:1.

**Decision for the user:** read the SI (reactant purity) and then either keep the bound one-sided or score 12.2–13.2 two-sided.

## Candidates that need a PDF (numbers UNCONFIRMED; not in systems.yaml)
- **Cope rearrangement of meso- and (±)-3,4-dimethylhexa-1,5-diene** (chair vs boat). Doering & Roth, *Tetrahedron* 18, 67 (1962), [doi:10.1016/0040-4020(62)80025-8](https://doi.org/10.1016/0040-4020(62)80025-8). The best rearrangement benchmark: 22 atoms, closed-shell, concerted.
  - meso: `C=C[C@@H](C)[C@@H](C)C=C`; (±): `C=C[C@@H](C)[C@H](C)C=C`.
  - Products: (E,Z) `C/C=C/CC/C=C\C`, (E,E) `C/C=C/CC/C=C/C`, (Z,Z) `C/C=C\CC/C=C\C`.
  - Only a secondary source was reachable. A Caltech thesis (thesis.caltech.edu/5378) says *"On the basis of product ratios, Doering and Roth estimated that ΔΔG‡ (boat–chair) for meso 138a was 5.7 kcal/mol"* and labels the meso → (E,Z) product *"(99.7% yield)"*.
  - The textbook (±) → 90 % (E,E) + 10 % (Z,Z), the temperature and the phase were not found in any text read.
- **Claisen rearrangement of the four stereoisomeric propenyl but-2-enyl ethers.** Vittorelli, Hansen, Schmid, *Helv. Chim. Acta* 58, 1293 (1975), [doi:10.1002/hlca.19750580508](https://doi.org/10.1002/hlca.19750580508). Chair/boat diastereomer ratios, 20 atoms. Closed access.
- **Formic acid:**
  - Saito et al. 2005: the measured k2/k1 values.
  - Hsu et al. 1982: whether "~10" appears anywhere in the paper.
- **Methyl formate:**
  - Ren et al. 2013: k1 and the minor channels.
  - Peukert et al. 2012: k4 by H-ARAS.
  - Wenz et al. 2023: k4/(k1+k4).
  - Döntgen, Wildenberg, Heufer 2026 (CC-BY, [doi:10.1002/kin.70035](https://doi.org/10.1002/kin.70035)): new methyl formate and formic acid data; the publisher and repository refused automated download.
- **Cp + methyl acrylate solvent series:** Berson et al. 1962; Cativiela et al. 1994 (restricted copy at DIGITAL.CSIC).
- **Isoprene + methyl acrylate:**
  - Renslo et al., *J. Org. Chem.* 62, 4530 (1997): the peer-reviewed version of the thesis data.
  - Hennis 1963: temperature independence.
  - Inukai & Kojima 1966: AlCl3, 95:5.
- **Furan + maleimide:** Rulíšek et al. 2005 full text and SI (solvent, concentrations, kinetic scheme). MPG.PuRe item_442180_6 is MPG-network only.
- **Furan + maleic anhydride literature (secondary only):** Lee & Herndon, *J. Org. Chem.* 43, 518 (1978); Tobia et al., *J. Org. Chem.* 58, 6701 (1993).

## Candidates considered and not used
- **Ethanol pyrolysis** (Sivaramakrishnan 2010; Wu et al. 2011): radical channels dominate the comparison; numbers not verified.
- **Glyoxal** (Friedrichs 2008): radical HCO channel, density dependent.
- **Acetaldehyde, nitromethane:** roaming-dominated, which TST misses.
- **Phenyl + O2:** open-shell, multireference, primary and secondary products mixed.
- **Glucose/fructose → HMF, formose, Pd couplings, alkoxide eliminations:** salts, catalysts, ions or explicit water.
- **2-Methylfuran, furan:** radical-rich at shock-tube temperatures.
- **Cp + methyl vinyl ketone** (Lording Table 1: 79:21) and **Cp + acrylonitrile** (55:45): confirmed and usable as extra Diels-Alder rungs. They are left out to keep chemistry types balanced. 55:45 (0.14 kcal/mol) is too small to resolve.
- **Torquoselective opening of a 3-silylcyclobutene** (Murakami, Miyamoto, Ito, *J. Synth. Org. Chem. Jpn.* 60, 1049, 2002; cis:trans 69:31, m-xylene, 140 °C): primary and open, but 41 atoms with Si.
- **Retro-Diels-Alder kinetics of furan adducts** (Widstrom & Lear 2019): rate-constant units not stated and the isomer not assigned.
- **Neat furan + maleic anhydride** (Lennox et al. 2026): the reaction mixture solidifies, so it is not a solution or gas benchmark.

## How the texts were obtained
Downloads are in the curator's scratchpad, not committed. Scripts are in `data/runs/explorer-team/curator/scripts/`.
- Elwardany 2015: KAUST accepted manuscript via the Wayback Machine.
- Cho 2024: OSTI accepted manuscript.
- Lording 2020: Europe PMC full-text XML.
- Saito 2005, Wenz 2023: PubMed abstracts via NCBI E-utilities.
- Saito 1984, Cativiela 1994: OpenAlex abstracts.
- Hsu 1982: NYCU research portal abstract.
- Chang 2007: NYCU repository PDF via the DSpace REST API.

Sites with bot walls (KAUST, Wiley, RWTH, DIGITAL.CSIC, RSC, ACS) were not circumvented.
