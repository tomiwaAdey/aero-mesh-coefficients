# Learning Aerodynamic Mesh Semantics from Triangulated Lifting Surfaces

Paper and reference implementation by **Tomiwa Adey**, University of Bristol. Research collaboration and supervision by **Ann Gaitonde**.

[Read the paper](paper/learning-aerodynamic-mesh-semantics.pdf) · [Publication page](https://www.tomiwaadey.com/publications/learning-aerodynamic-mesh-semantics)

## Abstract

This study presents a method for converting an unstructured triangulated lifting surface into a structured aerodynamic lattice from which aerodynamic coefficients can be calculated. The input geometry does not identify its leading edge, trailing edge, tips or camber surface. It is therefore repaired, placed in a consistent frame, and represented by 19 local geometrical and topological quantities for each edge. Classical classifiers assign one of four aerodynamic meanings to every edge. Connected boundary paths are then recovered, the upper and lower surfaces are paired, and a quadrilateral vortex lattice is constructed.

Models are selected using validation geometries and assessed using complete physical geometries. Four separate triangulations and three mesh densities are used so that recognition of the wing can be distinguished from recognition of one mesh pattern. The validation-selected classifiers achieved mean geometry-level macro F1 scores of 0.9994, 0.9976 and 0.9987 on the IID, unseen-mesher and cranked-wing-family tests. The fixed geometrical rule achieved 0.7280, 0.6454 and 0.7379 respectively.

On paired valid cases, mean absolute lift-coefficient error fell from 0.0473 to 0.0050 for IID geometry, from 0.1036 to 0.0086 for the unseen mesher, and from 0.0476 to 0.0073 for the cranked-wing family. Across 12 matched rectangular-wing cases, the internal vortex-lattice solver differed from AVL 3.32 by mean absolute values of 0.01771 in CL, 0.00053 in CDi and 0.00123 in CM.

Learned-only reconstruction produced a valid lattice for 1/5 ONERA M6 model seeds and 1/5 NASA CRM model seeds. The validity-gated hybrid produced a valid lattice in all 10/10 external runs by using the adaptive geometrical fallback when required. The method is therefore reliable over the generated geometry classes and independent triangulations considered here, but learned boundary recognition does not transfer reliably to either external geometry without a validity gate.

## Implementation

The implementation follows the paper directly:

```text
triangulated lifting surface
  -> mesh repair and canonical frame
  -> edge descriptors and semantic classification
  -> connected aerodynamic boundaries
  -> structured quadrilateral lattice
  -> vortex-lattice coefficients
```

- `src/aero_mesh/` contains geometry processing, classification, reconstruction, aerodynamic solvers and wake calculations.
- `experiments/` contains the evaluation stages reported in the paper.
- `tests/` covers geometry, reconstruction, solver conventions, external assets and reference tools.
- `paper/` contains the manuscript, bibliography, figures and published PDF.
- `data/source-manifest.json` records the external reference artifacts and their checksums.

Generated datasets, fitted models and experiment outputs are written to `results/` and are intentionally not versioned.

## Reproduce

The complete run uses Python 3.12, Docker, Pandoc and Tectonic.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[cad]'
make reproduce PYTHON=.venv/bin/python
```

To run the test suite without regenerating the experiments:

```bash
make test PYTHON=.venv/bin/python
```
