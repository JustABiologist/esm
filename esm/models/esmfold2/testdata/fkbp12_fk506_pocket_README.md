# ESMFold2 pocket-conditioning test fixture

Test case: human FKBP12 bound to FK506 derivative FK5.

- PDB: `1FKJ`
- Title: atomic structure of FKBP12-FK506 complex
- Resolution: 1.7 A
- Protein: chain `A`, 107 aa
- Ligand: chain `L` in the local fixture, CCD `FK5`
- Contact definition: non-hydrogen protein-ligand atom distance <= 4.0 A in
  the RCSB `1FKJ` coordinates.

The `PocketConditioning.contacts` fixture uses zero-based sequence residue
indices to match existing ESMFold2 residue-index conventions. The expected
output file also records the corresponding one-based PDB residue numbers.

The fast preparation tests do not require model weights, GPU, or network
access. The acceptance test **does** run local inference with ESMFold2 and
should be invoked explicitly:

```bash
ESMFOLD2_RUN_INFERENCE=1 pytest -q esm/models/esmfold2/pocket_conditioning_test.py
```

The inference test uses local code plus Hugging Face model loading via
`ESMFold2Model.from_pretrained("biohub/ESMFold2")`; it is not a Biohub hosted
API test.
