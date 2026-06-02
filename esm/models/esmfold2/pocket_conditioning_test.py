"""Target tests for experimental ESMFold2 ligand pocket conditioning.

These tests intentionally avoid loading model weights. They verify that
PocketConditioning is a real local input-preparation feature, not just a
serializable field.
"""

from __future__ import annotations

import ast
import gc
import importlib
import inspect
import json
import os
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.processor import clean_esmfold2_input
from esm.utils.structure.input_builder import (
    LigandInput,
    PocketConditioning,
    ProteinInput,
    StructurePredictionInput,
    deserialize_structure_prediction_input,
    serialize_structure_prediction_input,
)

FIXTURE_DIR = Path(__file__).with_name("testdata")
INPUT_FIXTURE = FIXTURE_DIR / "fkbp12_fk506_pocket_input.json"
EXPECTED_FIXTURE = FIXTURE_DIR / "fkbp12_fk506_expected_output.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "esmfold2_pocket_1fkj"


def _release_torch_memory() -> None:
    """Drop Python references first, then ask active Torch backends to release caches."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    mps = getattr(torch, "mps", None)
    if (
        os.environ.get("ESMFOLD2_DEVICE") == "mps"
        and mps is not None
        and hasattr(mps, "empty_cache")
    ):
        try:
            mps.empty_cache()
        except Exception:
            pass

    gc.collect()


def _load_fixture() -> tuple[StructurePredictionInput, dict]:
    raw_input = json.loads(INPUT_FIXTURE.read_text())
    expected = json.loads(EXPECTED_FIXTURE.read_text())
    spi = deserialize_structure_prediction_input(raw_input["structure_prediction_input"])
    return spi, expected


def _split_pocket_contact(contact: tuple) -> tuple[str, int, float]:
    assert len(contact) == 3, (
        "Pocket contacts must be residue-level distance hints: "
        "(chain_id, residue_index_zero_based, target_distance_angstrom)."
    )
    chain_id, residue_index, target_distance = contact
    return str(chain_id), int(residue_index), float(target_distance)


def _artifact_output_dir() -> Path:
    return Path(os.environ.get("ESMFOLD2_TEST_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


def _chain_token_indices(complex_obj, chain_id: str) -> list[int]:
    return [
        i
        for i, numeric_chain_id in enumerate(complex_obj.chain_id.tolist())
        if complex_obj.metadata.chain_lookup.get(int(numeric_chain_id)) == chain_id
    ]


def _token_atom_positions(complex_obj, token_index: int) -> np.ndarray:
    start, end = complex_obj.token_to_atoms[token_index]
    return complex_obj.atom_positions[int(start) : int(end)]


def _pocket_distance_metrics(complex_obj, pocket: PocketConditioning) -> dict:
    ligand_tokens = _chain_token_indices(complex_obj, pocket.binder_chain_id)
    ligand_atoms = (
        np.concatenate([_token_atom_positions(complex_obj, idx) for idx in ligand_tokens])
        if ligand_tokens
        else np.zeros((0, 3), dtype=np.float32)
    )
    contacts = []
    for raw_contact in pocket.contacts:
        contact_chain_id, residue_index, target_distance = _split_pocket_contact(raw_contact)
        contact_tokens = _chain_token_indices(complex_obj, contact_chain_id)
        predicted_distance = None
        if residue_index < len(contact_tokens) and len(ligand_atoms) > 0:
            residue_atoms = _token_atom_positions(complex_obj, contact_tokens[residue_index])
            distances = np.linalg.norm(
                residue_atoms[:, None, :] - ligand_atoms[None, :, :], axis=-1
            )
            predicted_distance = float(np.min(distances))
        contacts.append(
            {
                "chain_id": contact_chain_id,
                "residue_index_zero_based": residue_index,
                "target_distance_angstrom": target_distance,
                "predicted_min_distance_angstrom": predicted_distance,
                "absolute_error_angstrom": (
                    abs(predicted_distance - target_distance)
                    if predicted_distance is not None
                    else None
                ),
            }
        )

    finite_errors = [
        item["absolute_error_angstrom"]
        for item in contacts
        if item["absolute_error_angstrom"] is not None
    ]
    return {
        "distance_targets_are_soft_hints": True,
        "contacts": contacts,
        "mean_absolute_error_angstrom": (
            float(np.mean(finite_errors)) if finite_errors else None
        ),
        "max_absolute_error_angstrom": (
            float(np.max(finite_errors)) if finite_errors else None
        ),
    }


def _write_inference_artifacts(label: str, result, pocket: PocketConditioning) -> dict:
    output_dir = _artifact_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    mmcif = result.complex.to_mmcif()
    cif_path = output_dir / f"fkbp12_fk506_{label}.cif"
    metrics_path = output_dir / f"fkbp12_fk506_{label}_metrics.json"
    metrics = _pocket_distance_metrics(result.complex, pocket)

    cif_path.write_text(mmcif)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return {"mmcif": str(cif_path), "metrics": str(metrics_path), **metrics}


@pytest.fixture
def fkbp12_fk506_input() -> tuple[StructurePredictionInput, dict]:
    return _load_fixture()


@pytest.fixture(autouse=True)
def release_memory_after_each_test() -> None:
    yield
    _release_torch_memory()


@pytest.fixture(autouse=True)
def offline_ccd(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Keep tests offline by stubbing CCD lookups used during tokenization."""
    if request.node.get_closest_marker("esmfold2_inference"):
        return

    import esm.models.esmfold2.prepare_input as prep

    ligand_atoms = [
        ("C1", "C", 0),
        ("C2", "C", 0),
        ("O1", "O", 0),
    ]

    def fake_get_ligand_ccd_atoms_with_charges(comp_id: str):
        if comp_id != "FK5":
            return None
        return ligand_atoms

    def fake_get_ligand_idealized_atom_pos(comp_id: str, atom_name: str):
        positions = {
            "C1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "C2": np.array([1.5, 0.0, 0.0], dtype=np.float32),
            "O1": np.array([0.0, 1.2, 0.0], dtype=np.float32),
        }
        return positions.get(atom_name)

    monkeypatch.setattr(prep, "get_idealized_atom_pos", lambda *_: None)
    monkeypatch.setattr(
        prep,
        "get_ligand_ccd_atoms_with_charges",
        fake_get_ligand_ccd_atoms_with_charges,
    )
    monkeypatch.setattr(
        prep,
        "get_ligand_idealized_atom_pos",
        fake_get_ligand_idealized_atom_pos,
    )
    monkeypatch.setattr(prep, "get_ccd_leaving_atoms", lambda *_: set())
    monkeypatch.setattr(
        prep, "get_ligand_ccd_bonds", lambda comp_id: [("C1", "C2"), ("C1", "O1")]
    )


def test_pocket_conditioning_is_public_esmfold2_api() -> None:
    esmfold2 = importlib.import_module("esm.models.esmfold2")
    assert hasattr(esmfold2, "PocketConditioning")


def test_fkbp12_fk506_pocket_fixture_round_trips(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, expected = fkbp12_fk506_input
    assert spi.pocket is not None
    assert spi.pocket.binder_chain_id == expected["ligand_chain_id"]
    contacts = [_split_pocket_contact(contact) for contact in spi.pocket.contacts]
    assert [idx for _, idx, _ in contacts] == expected[
        "contact_residue_indices_zero_based"
    ]
    assert [distance for _, _, distance in contacts] == pytest.approx(
        expected["contact_target_distances_angstrom"]
    )

    restored = deserialize_structure_prediction_input(
        json.loads(json.dumps(serialize_structure_prediction_input(spi)))
    )

    assert restored.pocket is not None
    assert restored.pocket.binder_chain_id == "L"
    assert restored.pocket.contacts == spi.pocket.contacts


def test_pocket_conditioning_input_is_residue_level_not_atom_level(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, expected = fkbp12_fk506_input
    assert spi.pocket is not None
    serialized_pocket = serialize_structure_prediction_input(spi)["pocket"]

    assert "atom" not in json.dumps(serialized_pocket).lower()
    assert len(spi.pocket.contacts) == len(expected["contact_evidence"])
    for raw_contact in spi.pocket.contacts:
        chain_id, residue_index, target_distance = _split_pocket_contact(raw_contact)
        assert chain_id == expected["protein_chain_id"]
        assert residue_index >= 0
        assert 2.0 <= target_distance <= 6.0

    evidence_distances = [
        item["target_distance_angstrom"] for item in expected["contact_evidence"]
    ]
    assert [distance for _, _, distance in spi.pocket.contacts] == pytest.approx(
        evidence_distances
    )


def test_clean_esmfold2_input_preserves_pocket_conditioning(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    cleaned = clean_esmfold2_input(spi)

    assert cleaned.pocket is not None
    assert cleaned.pocket.binder_chain_id == spi.pocket.binder_chain_id
    assert cleaned.pocket.contacts == spi.pocket.contacts


def test_prepare_input_materializes_pocket_conditioning_features(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, expected = fkbp12_fk506_input
    features, chains = prepare_esmfold2_input(spi)
    chain_by_id = {chain.chain_id: chain for chain in chains}

    assert expected["protein_chain_id"] in chain_by_id
    assert expected["ligand_chain_id"] in chain_by_id
    for key in expected["expected_feature_behavior"]["required_feature_keys"]:
        assert key in features

    protein_asym = chain_by_id[expected["protein_chain_id"]].asym_id
    ligand_asym = chain_by_id[expected["ligand_chain_id"]].asym_id
    contact_residue_indices = set(expected["contact_residue_indices_zero_based"])

    target_token_indices = [
        i
        for i, (asym_id, residue_index) in enumerate(
            zip(features["asym_id"].tolist(), features["residue_index"].tolist())
        )
        if asym_id == protein_asym and residue_index in contact_residue_indices
    ]
    ligand_token_indices = [
        i for i, asym_id in enumerate(features["asym_id"].tolist()) if asym_id == ligand_asym
    ]

    assert len(target_token_indices) == len(contact_residue_indices)
    assert ligand_token_indices

    pocket_feature = features["pocket_feature"]
    assert pocket_feature[target_token_indices].sum().item() == len(target_token_indices)

    cross_mask = features["disto_cond_mask"][target_token_indices][:, ligand_token_indices]
    assert cross_mask.any().item()


def test_prepare_input_encodes_per_residue_soft_distance_hints(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, expected = fkbp12_fk506_input
    assert spi.pocket is not None
    features, chains = prepare_esmfold2_input(spi)
    chain_by_id = {chain.chain_id: chain for chain in chains}
    protein_asym = chain_by_id[expected["protein_chain_id"]].asym_id
    ligand_asym = chain_by_id[expected["ligand_chain_id"]].asym_id
    ligand_token_indices = [
        i for i, asym_id in enumerate(features["asym_id"].tolist()) if asym_id == ligand_asym
    ]

    observed_bins = []
    for raw_contact, expected_bin in zip(
        spi.pocket.contacts, expected["contact_target_distance_bins"]
    ):
        contact_chain_id, residue_index, _target_distance = _split_pocket_contact(raw_contact)
        assert contact_chain_id == expected["protein_chain_id"]
        target_token_indices = [
            i
            for i, (asym_id, token_residue_index) in enumerate(
                zip(features["asym_id"].tolist(), features["residue_index"].tolist())
            )
            if asym_id == protein_asym and token_residue_index == residue_index
        ]
        assert len(target_token_indices) == 1
        contact_token_index = target_token_indices[0]
        contact_mask = features["disto_cond_mask"][contact_token_index, ligand_token_indices]
        contact_bins = features["disto_cond"][contact_token_index, ligand_token_indices]
        assert contact_mask.all().item()
        assert set(contact_bins.tolist()) == {expected_bin}
        observed_bins.append(expected_bin)

    assert min(observed_bins) > 0
    assert len(set(observed_bins)) > 1


def test_invalid_pocket_binder_chain_id_raises(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    bad = replace(
        spi,
        pocket=PocketConditioning(binder_chain_id="Z", contacts=spi.pocket.contacts),
    )

    with pytest.raises(ValueError, match="binder_chain_id.*Z|Z.*binder_chain_id"):
        prepare_esmfold2_input(bad)


def test_invalid_pocket_contact_chain_id_raises(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    bad = replace(
        spi,
        pocket=PocketConditioning(binder_chain_id="L", contacts=[("Z", 25, 3.5)]),
    )

    with pytest.raises(ValueError, match="contact.*Z|Z.*contact|chain.*Z|Z.*chain"):
        prepare_esmfold2_input(bad)


def test_invalid_pocket_contact_distance_raises(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    bad = replace(
        spi,
        pocket=PocketConditioning(binder_chain_id="L", contacts=[("A", 25, -1.0)]),
    )

    with pytest.raises(ValueError, match="distance|angstrom|positive|non-negative"):
        prepare_esmfold2_input(bad)


def test_input_without_pocket_keeps_empty_pocket_features(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    spi, _ = fkbp12_fk506_input
    no_pocket = StructurePredictionInput(sequences=spi.sequences)
    features, _ = prepare_esmfold2_input(no_pocket)

    assert features["pocket_feature"].sum().item() == 0
    assert features["disto_cond_mask"].sum().item() == 0


def test_esmfold2_forward_explicitly_accepts_pocket_conditioning_inputs() -> None:
    """The model must not silently swallow conditioning tensors via **kwargs."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    signature = inspect.signature(ESMFold2Model.forward)
    for name in ("pocket_feature", "disto_cond", "disto_cond_mask"):
        assert name in signature.parameters, (
            f"ESMFold2Model.forward must explicitly accept {name!r}; "
            "otherwise local pocket conditioning can be silently ignored."
        )
        assert signature.parameters[name].kind is not inspect.Parameter.VAR_KEYWORD


def test_esmfold2_forward_source_uses_pocket_conditioning_inputs() -> None:
    """Static guardrail: explicit inputs must be read in the forward body."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    tree = ast.parse(textwrap.dedent(inspect.getsource(ESMFold2Model.forward)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    body_names = {
        node.id
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }

    for name in ("pocket_feature", "disto_cond", "disto_cond_mask"):
        assert name in body_names, (
            f"ESMFold2Model.forward declares {name!r} but does not read it "
            "in the executable body."
        )


def test_esmfold2_pocket_conditioning_is_not_fixed_hidden_state_ramp() -> None:
    """Pocket hints should use model-owned conditioning, not an arbitrary constant ramp."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    source = textwrap.dedent(inspect.getsource(ESMFold2Model.forward))
    forbidden_snippets = (
        "pocket_basis",
        "pocket_pair_bias",
        "torch.linspace(",
    )

    for snippet in forbidden_snippets:
        assert snippet not in source, (
            "Pocket conditioning must not be implemented as a hard-coded hidden-state "
            f"ramp/bias; found {snippet!r} in ESMFold2Model.forward."
        )


def test_local_runner_does_not_patch_transformers_site_packages_at_runtime() -> None:
    """The model change must live in repo-owned code, not mutable site-packages text edits."""
    runner = REPO_ROOT / "tools" / "run_esmfold2_pocket_goal_tests.sh"
    runner_text = runner.read_text()

    assert "patch_esmfold2_pocket_conditioning.py" not in runner_text
    assert "site-packages" not in runner_text
    assert not (REPO_ROOT / "tools" / "patch_esmfold2_pocket_conditioning.py").exists()


@pytest.mark.esmfold2_inference
@pytest.mark.skipif(
    os.environ.get("ESMFOLD2_RUN_INFERENCE") != "1",
    reason="Set ESMFOLD2_RUN_INFERENCE=1 to run the local ESMFold2 inference test.",
)
def test_local_esmfold2_inference_with_fkbp12_fk506_pocket(
    tmp_path: Path,
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    """Run a real local ESMFold2 inference smoke test with pocket conditioning."""
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from esm.models.esmfold2 import ESMFold2InputBuilder

    spi, expected = fkbp12_fk506_input
    model_id = os.environ.get("ESMFOLD2_MODEL_ID", "biohub/ESMFold2")
    device_name = os.environ.get(
        "ESMFOLD2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "ESMFOLD2_DEVICE=cuda but CUDA is unavailable"

    model = builder = result = None
    try:
        model = ESMFold2Model.from_pretrained(model_id).to(device).eval()
        builder = ESMFold2InputBuilder(ccd_cache=os.environ.get("ESMFOLD2_CCD_CACHE"))

        result = builder.fold(
            model,
            spi,
            num_loops=int(os.environ.get("ESMFOLD2_TEST_NUM_LOOPS", "1")),
            num_sampling_steps=int(os.environ.get("ESMFOLD2_TEST_NUM_SAMPLING_STEPS", "4")),
            num_diffusion_samples=1,
            seed=0,
            complex_id="1FKJ_FK5_pocket_test",
        )

        mmcif = result.complex.to_mmcif()
        out_path = tmp_path / "fkbp12_fk506_pocket.cif"
        out_path.write_text(mmcif)

        assert "_atom_site." in mmcif
        assert expected["expected_inference_behavior"]["mmcif_must_contain_ligand_ccd"] in mmcif
        assert len(mmcif) > 1000
    finally:
        del result
        del builder
        del model
        _release_torch_memory()


@pytest.mark.esmfold2_inference
@pytest.mark.skipif(
    os.environ.get("ESMFOLD2_RUN_INFERENCE") != "1",
    reason="Set ESMFOLD2_RUN_INFERENCE=1 to run the local ESMFold2 inference test.",
)
def test_local_esmfold2_inference_changes_when_pocket_conditioning_changes(
    fkbp12_fk506_input: tuple[StructurePredictionInput, dict],
) -> None:
    """Same seed/input should differ when pocket conditioning is removed."""
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from esm.models.esmfold2 import ESMFold2InputBuilder

    spi, _ = fkbp12_fk506_input
    no_pocket = StructurePredictionInput(sequences=spi.sequences)

    model_id = os.environ.get("ESMFOLD2_MODEL_ID", "biohub/ESMFold2")
    device_name = os.environ.get(
        "ESMFOLD2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "ESMFOLD2_DEVICE=cuda but CUDA is unavailable"

    model = builder = pocket_features = no_pocket_features = None
    pocket_chains = no_pocket_chains = None
    pocket_result = no_pocket_result = None
    pocket_output = no_pocket_output = None
    manifest = {
        "source_pdb_id": "1FKJ",
        "distance_targets_are_soft_hints": True,
        "conditioned": None,
        "unconditioned": None,
    }
    try:
        model = ESMFold2Model.from_pretrained(model_id).to(device).eval()
        builder = ESMFold2InputBuilder(ccd_cache=os.environ.get("ESMFOLD2_CCD_CACHE"))
        pocket_features, pocket_chains = builder.prepare_input(spi, seed=0, device=device)
        no_pocket_features, no_pocket_chains = builder.prepare_input(
            no_pocket, seed=0, device=device
        )

        assert pocket_features["disto_cond_mask"].any()
        assert not no_pocket_features["disto_cond_mask"].any()

        forward_kwargs = dict(
            num_loops=int(os.environ.get("ESMFOLD2_TEST_NUM_LOOPS", "1")),
            num_sampling_steps=int(os.environ.get("ESMFOLD2_TEST_NUM_SAMPLING_STEPS", "4")),
            num_diffusion_samples=1,
            early_exit=False,
        )

        with torch.inference_mode():
            torch.manual_seed(123)
            pocket_output = model(**pocket_features, **forward_kwargs)
            torch.manual_seed(123)
            no_pocket_output = model(**no_pocket_features, **forward_kwargs)

        changed = False
        for key in ("distogram_logits", "sample_atom_coords"):
            assert key in pocket_output
            assert key in no_pocket_output
            changed = changed or not torch.allclose(
                pocket_output[key], no_pocket_output[key], atol=1e-5, rtol=1e-5
            )

        assert changed, (
            "Pocket conditioning had no numerical effect on local ESMFold2 inference. "
            "This usually means the tensors were prepared but not consumed by the model."
        )

        assert spi.pocket is not None
        pocket_result = builder.decode(
            pocket_output,
            pocket_features,
            pocket_chains,
            num_diffusion_samples=1,
            complex_id="1FKJ_FK5_conditioned",
        )
        no_pocket_result = builder.decode(
            no_pocket_output,
            no_pocket_features,
            no_pocket_chains,
            num_diffusion_samples=1,
            complex_id="1FKJ_FK5_unconditioned",
        )
        if isinstance(pocket_result, list):
            pocket_result = pocket_result[0]
        if isinstance(no_pocket_result, list):
            no_pocket_result = no_pocket_result[0]

        manifest["conditioned"] = _write_inference_artifacts(
            "conditioned", pocket_result, spi.pocket
        )
        manifest["unconditioned"] = _write_inference_artifacts(
            "unconditioned", no_pocket_result, spi.pocket
        )
        manifest_path = _artifact_output_dir() / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        for label in ("conditioned", "unconditioned"):
            output = manifest[label]
            assert output is not None
            assert Path(output["mmcif"]).exists()
            assert Path(output["metrics"]).exists()
            assert len(output["contacts"]) == len(spi.pocket.contacts)
            assert output["distance_targets_are_soft_hints"] is True
            assert any(
                item["predicted_min_distance_angstrom"] is not None
                for item in output["contacts"]
            )
    finally:
        del manifest
        del no_pocket_result
        del pocket_result
        del no_pocket_chains
        del pocket_chains
        del no_pocket_output
        del pocket_output
        del no_pocket_features
        del pocket_features
        del builder
        del model
        _release_torch_memory()
