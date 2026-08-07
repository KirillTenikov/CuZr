# Validation status

Status labels:

- **existing** — represented in the submitted project;
- **implemented** — code exists in `revision/`, but may still need A100 validation;
- **planned** — agreed scientifically, not yet implemented;
- **excluded** — deliberately outside the revision.

| Validation block | Submitted project | Revision code v0.2 | Next scientific action | Status |
|---|---|---|---|---|
| Global energy/force test RMSE | yes | not changed | reproduce and extend by subsets | existing |
| Learning curves | limited/final metrics | no extractor yet | implement MACE and ACE convergence analysis | planned |
| State-resolved DFT errors | no | no | inspect dataset metadata, classify only supported states | planned |
| Composition-resolved DFT errors | no | no | implement from atomic species | planned |
| DFT stress errors | not reported | no | detect stress-labelled configurations and report sample size | planned |
| FCC/HCP/B2 EOS | yes | original scripts retained | retain and verify only as needed | existing |
| Vacancy energies | yes | original scripts retained | retain without major expansion | existing |
| B2 formation energy | no | no | add inexpensive static calculation | planned |
| Production-quality glass preparation | no | stage generator present | run and inspect first A100 pilot | implemented |
| Three independent realizations | no | matrix generator supports seeds 43–45 | freeze initial densities and execute | implemented |
| Tail pressure/density statistics | no | thermo summarizer present | validate parser against A100 logs | implemented |
| Fixed-cell inherent structure | partial/implicit | explicit stage present | compare with finite-temperature state | implemented |
| Box-relaxed inherent structure | no | explicit optional stage present | decide whether all runs or selected runs | implemented |
| Total RDF | yes | implemented | compare old and revised preparation | implemented |
| Partial RDFs | no | implemented | validate normalization and integration limits | implemented |
| Coordination numbers | no | preliminary implementation | visually validate first minima | implemented |
| Weighted structure factor S(q) | no | no | implement after choosing experiment and weights | planned |
| Local-order/Voronoi analysis | no | no | add compact icosahedral metric | planned |
| 1024-versus-4000 comparison | no | runner supports arbitrary atom count | define selected matrix and execute | partially implemented |
| Short NVE stability | only technical launch checks | opt-in stage present | run after preparation pilot | implemented |
| Elastic screening, all models | no | no | implement K/G strain workflow | planned |
| Detailed amorphous elasticity | no | no | run for top 2–3 after screening | planned |
| Crystalline elastic constants | no/full tensor absent | no | optional after amorphous workflow | optional |
| Viscosity | no | none | do not add | excluded |
| Diffusion campaign | no | none | do not add | excluded |
| Glass-transition temperature | no | none | do not add | excluded |
| Melting/phase diagram | no | none | do not add | excluded |
| New DFT calculations | no | none | use existing held-out DFT data | excluded |

## Immediate milestone

The project is currently at the transition from **implemented input-generation framework** to **first physical A100 validation**. The next accepted milestone is not the full matrix; it is one trustworthy MACE_C pilot with complete logs, summaries and runtime notes.
