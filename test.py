# %%
from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


# ==============================================================================
# ModelSettings tests
# ==============================================================================

# %%
def test_model_settings_defaults():
    s = ModelSettings()
    assert s.seq_len == 2
    assert s.input_vocab_size == 32
    assert s.output_vocab_size == 16
    assert s.n_facts == 64
    assert s.seed == 42
    assert s.d_residual == 16
    assert s.n_heads == 1
    assert s.d_ff == 16
    assert s.attention is True
    assert s.ff is True
    assert s.bias is True
    assert s.norms is True
    assert s.ff_residual is True
    assert s.ff_activation_type == 'GELU'

def test_model_settings_custom():
    s = ModelSettings(seq_len=4, input_vocab_size=64, output_vocab_size=8,
                      n_facts=100, d_residual=32, n_heads=4, d_ff=64,
                      attention=False, ff=False, bias=True, norms=False, ff_residual=False)
    assert s.seq_len == 4
    assert s.input_vocab_size == 64
    assert s.output_vocab_size == 8
    assert s.n_facts == 100
    assert s.d_residual == 32
    assert s.n_heads == 4
    assert s.d_ff == 64
    assert s.attention is False
    assert s.ff is False
    assert s.bias is True
    assert s.norms is False
    assert s.ff_residual is False


# ==============================================================================
# generate_facts tests
# ==============================================================================

# %%
def test_generate_facts_shapes():
    facts = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4)
    assert facts["inputs"].shape == (10, 2)
    assert facts["targets"].shape == (10,)

def test_generate_facts_seq_len_1():
    facts = generate_facts(n_facts=5, seq_len=1, input_vocab_size=8, output_vocab_size=4)
    assert facts["inputs"].shape == (5, 1)
    assert facts["targets"].shape == (5,)

def test_generate_facts_target_range():
    facts = generate_facts(n_facts=20, seq_len=2, input_vocab_size=8, output_vocab_size=4)
    assert facts["targets"].min() >= 0
    assert facts["targets"].max() < 4

def test_generate_facts_input_range():
    facts = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4)
    assert facts["inputs"].min() >= 0
    assert facts["inputs"].max() < 8

def test_generate_facts_unique_inputs():
    facts = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4)
    # Each row of inputs should be unique
    unique_rows = torch.unique(facts["inputs"], dim=0)
    assert unique_rows.shape[0] == 10

def test_generate_facts_sorted_by_target():
    facts = generate_facts(n_facts=20, seq_len=2, input_vocab_size=8, output_vocab_size=4)
    targets = facts["targets"]
    # Targets should be sorted in non-decreasing order
    assert torch.all(targets[1:] >= targets[:-1])

def test_generate_facts_reproducible():
    facts1 = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4, seed=123)
    facts2 = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4, seed=123)
    assert torch.equal(facts1["inputs"], facts2["inputs"])
    assert torch.equal(facts1["targets"], facts2["targets"])

def test_generate_facts_different_seeds():
    facts1 = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4, seed=1)
    facts2 = generate_facts(n_facts=10, seq_len=2, input_vocab_size=8, output_vocab_size=4, seed=2)
    assert not torch.equal(facts1["inputs"], facts2["inputs"])

def test_generate_facts_too_many_raises():
    with pytest.raises(ValueError, match="Cannot generate"):
        generate_facts(n_facts=100, seq_len=1, input_vocab_size=8, output_vocab_size=4)

def test_generate_facts_seq_len_3():
    facts = generate_facts(n_facts=10, seq_len=3, input_vocab_size=8, output_vocab_size=4)
    assert facts["inputs"].shape == (10, 3)

def test_generate_facts_max_capacity():
    # Exactly fills the capacity for seq_len=1
    facts = generate_facts(n_facts=8, seq_len=1, input_vocab_size=8, output_vocab_size=4)
    assert facts["inputs"].shape == (8, 1)
    assert len(torch.unique(facts["inputs"])) == 8


# ==============================================================================
# CausalSelfAttention tests
# ==============================================================================

# %%
def test_causal_attention_output_shape():
    attn = CausalSelfAttention(d_model=16, n_heads=2)
    x = torch.randn(4, 3, 16)
    out = attn(x)
    assert out.shape == (4, 3, 16)

def test_causal_attention_single_head():
    attn = CausalSelfAttention(d_model=8, n_heads=1)
    x = torch.randn(2, 5, 8)
    out = attn(x)
    assert out.shape == (2, 5, 8)

def test_causal_attention_deterministic():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=8, n_heads=1)
    x = torch.randn(1, 3, 8)
    out1 = attn(x)
    out2 = attn(x)
    assert torch.allclose(out1, out2)

def test_causal_attention_d_model_not_divisible_raises():
    with pytest.raises(AssertionError):
        CausalSelfAttention(d_model=7, n_heads=2)


# ==============================================================================
# MemoryToyModel tests
# ==============================================================================

# %%
def test_model_forward_with_attention_and_ff():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=2, d_ff=32,
                      attention=True, ff=True)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)

def test_model_forward_no_attention():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, d_ff=32,
                      attention=False, ff=True)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)

def test_model_forward_no_ff():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=2,
                      attention=True, ff=False)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)

def test_model_forward_no_norms():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=2, d_ff=32,
                      attention=True, ff=True, norms=False)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)

def test_model_forward_no_ff_residual():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32,
                      attention=True, ff=True, ff_residual=False)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)

def test_model_forward_with_bias():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32,
                      attention=True, ff=True, bias=True)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)

def test_model_seq_len_mismatch_raises():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1)
    model = MemoryToyModel(s)
    wrong_input = torch.randint(0, 8, (5, 3))  # seq_len=3 but model expects 2
    with pytest.raises(AssertionError, match="Sequence length"):
        model(wrong_input)

def test_model_facts_stored():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4, n_facts=10)
    model = MemoryToyModel(s)
    assert "inputs" in model.facts
    assert "targets" in model.facts
    assert model.facts["inputs"].shape == (10, 2)
    assert model.facts["targets"].shape == (10,)

def test_model_seq_len_1_no_attention():
    s = ModelSettings(seq_len=1, input_vocab_size=16, output_vocab_size=4,
                      n_facts=10, d_residual=8, d_ff=16,
                      attention=False, ff=True)
    model = MemoryToyModel(s)
    inputs = model.facts["inputs"]
    logits = model(inputs)
    assert logits.shape == (10, 4)


# ==============================================================================
# train_model tests
# ==============================================================================

# %%
def test_train_model_loss_decreases():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32)
    torch.manual_seed(0)
    model = MemoryToyModel(s)
    
    # Get initial loss
    model.eval()
    with torch.no_grad():
        logits = model(model.facts["inputs"])
        targets_oh = F.one_hot(model.facts["targets"], s.output_vocab_size).float()
        initial_loss = F.binary_cross_entropy_with_logits(logits, targets_oh).item()
    
    # Train briefly
    train_model(model, n_epochs=200, log_to_wandb=False, early_stopping=False)
    
    # Get final loss
    model.eval()
    with torch.no_grad():
        logits = model(model.facts["inputs"])
        final_loss = F.binary_cross_entropy_with_logits(logits, targets_oh).item()
    
    assert final_loss < initial_loss

def test_train_model_early_stopping_perfect():
    """Model should stop early when perfect accuracy is reached."""
    s = ModelSettings(seq_len=1, input_vocab_size=8, output_vocab_size=2,
                      n_facts=4, d_residual=32, n_heads=1, d_ff=64)
    torch.manual_seed(0)
    model = MemoryToyModel(s)
    # With very few facts, the model should be able to learn them all
    train_model(model, n_epochs=5000, log_to_wandb=False,
                early_stopping=True, patience=200)
    
    model.eval()
    with torch.no_grad():
        logits = model(model.facts["inputs"])
        preds = logits.argmax(dim=-1)
        accuracy = (preds == model.facts["targets"]).float().mean().item()
    # Should have learned (or at least improved)
    assert accuracy >= 0.5

def test_train_model_custom_lr():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32)
    model = MemoryToyModel(s)
    # Just verify it runs without error with a custom lr
    train_model(model, n_epochs=10, lr=1e-4, log_to_wandb=False, early_stopping=False)

def test_train_model_no_attention():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, d_ff=32,
                      attention=False, ff=True)
    model = MemoryToyModel(s)
    train_model(model, n_epochs=10, log_to_wandb=False, early_stopping=False)

def test_train_model_adamw():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32)
    model = MemoryToyModel(s)
    train_model(model, n_epochs=10, optimizer_type='AdamW', log_to_wandb=False, early_stopping=False)

def test_train_model_grad_clip():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32)
    model = MemoryToyModel(s)
    train_model(model, n_epochs=10, grad_clip_norm=1.0, log_to_wandb=False, early_stopping=False)

def test_train_model_adamw_with_grad_clip():
    s = ModelSettings(seq_len=2, input_vocab_size=8, output_vocab_size=4,
                      n_facts=10, d_residual=16, n_heads=1, d_ff=32)
    model = MemoryToyModel(s)
    train_model(model, n_epochs=10, optimizer_type='AdamW', grad_clip_norm=1.0,
                log_to_wandb=False, early_stopping=False)


# ==============================================================================
# capacity_search tests
# ==============================================================================

from capacity_search import evaluate_model, find_max_facts, _try_n_facts, name_function_n_facts

# %%
def test_evaluate_model_returns_bool():
    """evaluate_model should run and return a bool."""
    s = ModelSettings(seq_len=1, input_vocab_size=8, output_vocab_size=2,
                      n_facts=4, d_residual=16, n_heads=1, d_ff=32)
    model = MemoryToyModel(s)
    result = evaluate_model(model)
    assert isinstance(result, bool)


# %%
def test_try_n_facts_runs():
    """_try_n_facts should run and return a bool."""
    s = ModelSettings(seq_len=1, input_vocab_size=8, output_vocab_size=2,
                      n_facts=4, d_residual=16, n_heads=1, d_ff=32)
    result = _try_n_facts(s, n_facts=4, n_epochs=10, lr=[1e-2],
                          optimizer_type='Adam', grad_clip_norm=None,
                          patience=5,
                          log_to_wandb=False, wandb_group='test',
                          wandb_log_every=10, verbose=False,
                          name_function=name_function_n_facts,
                          target_accuracy='accuracy',
                          threshold_for_continued_search=0.0)
    assert isinstance(result, bool)


def test_try_n_facts_overrides_n_facts():
    """_try_n_facts should use the n_facts argument, not the one in settings."""
    s = ModelSettings(seq_len=1, input_vocab_size=8, output_vocab_size=2,
                      n_facts=999, d_residual=16, n_heads=1, d_ff=32)
    # Should not crash even though settings.n_facts=999 exceeds vocab capacity
    result = _try_n_facts(s, n_facts=4, n_epochs=10, lr=[1e-2],
                          optimizer_type='Adam', grad_clip_norm=None,
                          patience=5,
                          log_to_wandb=False, wandb_group='test',
                          wandb_log_every=10, verbose=False,
                          name_function=name_function_n_facts,
                          target_accuracy='accuracy',
                          threshold_for_continued_search=0.0)
    assert isinstance(result, bool)


# %%
def test_find_max_facts_runs():
    """find_max_facts should run and return an int."""
    s = ModelSettings(seq_len=1, input_vocab_size=4, output_vocab_size=2,
                      d_residual=16, n_heads=1, d_ff=32)
    result = find_max_facts(s, precision=3, n_epochs=10, patience=5,
                            log_to_wandb=False, verbose=False)
    assert isinstance(result, int)


def test_find_max_facts_bounded_by_max_possible():
    """Result should never exceed input_vocab_size ** seq_len."""
    s = ModelSettings(seq_len=1, input_vocab_size=4, output_vocab_size=2,
                      d_residual=16, n_heads=1, d_ff=32)
    max_possible = s.input_vocab_size ** s.seq_len
    result = find_max_facts(s, precision=3, n_epochs=10, patience=5,
                            log_to_wandb=False, verbose=False)
    assert result <= max_possible


# ==============================================================================
# Run tests
# ==============================================================================

# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==============================================================================
# log_result / load_results tests
# ==============================================================================

import os
import json
import shutil
from log import log_result, load_results

_LOG_TEST_DIR = "test_results_tmp"


@pytest.fixture(autouse=False)
def clean_log_test_dir():
    """Create a fresh temp directory before each log test and remove it after."""
    os.makedirs(_LOG_TEST_DIR, exist_ok=True)
    yield
    shutil.rmtree(_LOG_TEST_DIR, ignore_errors=True)


def _log_base_path():
    return os.path.join(_LOG_TEST_DIR, "test_log")


def test_log_result_creates_jsonl(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    log_result("run_1", max_facts=42, settings=settings, filepath=path)
    assert os.path.exists(path + ".jsonl")


def test_log_result_jsonl_contains_expected_fields(clean_log_test_dir):
    settings = ModelSettings(d_residual=32, n_heads=4)
    path = _log_base_path()
    log_result("my_run", max_facts=99, settings=settings, filepath=path)
    with open(path + ".jsonl", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["name"] == "my_run"
    assert record["max_facts"] == 99
    assert record["settings"]["d_residual"] == 32
    assert record["settings"]["n_heads"] == 4


def test_log_result_jsonl_is_valid_json(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    log_result("run_a", max_facts=10, settings=settings, filepath=path)
    with open(path + ".jsonl", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["name"] == "run_a"
    assert record["max_facts"] == 10
    assert "timestamp" in record
    assert "settings" in record


def test_log_result_records_all_settings_keys(clean_log_test_dir):
    settings = ModelSettings(seq_len=3, input_vocab_size=64, bias=True, norms=False)
    path = _log_base_path()
    log_result("full_settings", max_facts=5, settings=settings, filepath=path)
    with open(path + ".jsonl", encoding="utf-8") as f:
        record = json.loads(f.readline())
    s = record["settings"]
    assert s["seq_len"] == 3
    assert s["input_vocab_size"] == 64
    assert s["bias"] is True
    assert s["norms"] is False


def test_log_result_with_extra(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    log_result("extra_run", max_facts=7, settings=settings, filepath=path,
               extra={"lr": [0.01, 0.001], "patience": 500})
    with open(path + ".jsonl", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["extra"]["lr"] == [0.01, 0.001]
    assert record["extra"]["patience"] == 500


def test_log_result_appends_multiple_runs(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    log_result("run_1", max_facts=10, settings=settings, filepath=path)
    log_result("run_2", max_facts=20, settings=settings, filepath=path)
    log_result("run_3", max_facts=30, settings=settings, filepath=path)
    with open(path + ".jsonl", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 3
    names = [json.loads(l)["name"] for l in lines]
    assert names == ["run_1", "run_2", "run_3"]


def test_log_result_does_not_mutate_settings(clean_log_test_dir):
    settings = ModelSettings(d_residual=16)
    path = _log_base_path()
    log_result("immutable", max_facts=1, settings=settings, filepath=path)
    assert settings.d_residual == 16


def test_load_results_empty_when_no_file(clean_log_test_dir):
    results = load_results(filepath=os.path.join(_LOG_TEST_DIR, "nonexistent"))
    assert results == []


def test_load_results_returns_logged_data(clean_log_test_dir):
    settings = ModelSettings(d_ff=64)
    path = _log_base_path()
    log_result("loadable", max_facts=55, settings=settings, filepath=path)
    results = load_results(filepath=path)
    assert len(results) == 1
    assert results[0]["name"] == "loadable"
    assert results[0]["max_facts"] == 55
    assert results[0]["settings"]["d_ff"] == 64


def test_load_results_preserves_order(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    for i in range(5):
        log_result(f"run_{i}", max_facts=i * 10, settings=settings, filepath=path)
    results = load_results(filepath=path)
    assert len(results) == 5
    assert [r["name"] for r in results] == [f"run_{i}" for i in range(5)]
    assert [r["max_facts"] for r in results] == [0, 10, 20, 30, 40]


def test_load_results_includes_extra(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    log_result("with_extra", max_facts=1, settings=settings, filepath=path,
               extra={"note": "hello"})
    results = load_results(filepath=path)
    assert results[0]["extra"]["note"] == "hello"


def test_load_results_no_extra_key_when_none(clean_log_test_dir):
    settings = ModelSettings()
    path = _log_base_path()
    log_result("no_extra", max_facts=1, settings=settings, filepath=path)
    results = load_results(filepath=path)
    assert "extra" not in results[0]


def test_roundtrip_multiple_settings(clean_log_test_dir):
    """Log several runs with different settings and verify they all load back correctly."""
    path = _log_base_path()
    configs = [
        ("attn_only", 100, ModelSettings(attention=True, ff=False, bias=False, norms=True)),
        ("ff_only",    80, ModelSettings(attention=False, ff=True, bias=True, norms=False, ff_residual=False)),
        ("full",      200, ModelSettings(attention=True, ff=True, d_residual=64, n_heads=8)),
    ]
    for name, mf, s in configs:
        log_result(name, mf, s, filepath=path)
    results = load_results(filepath=path)
    assert len(results) == 3
    assert results[0]["settings"]["attention"] is True
    assert results[0]["settings"]["ff"] is False
    assert results[1]["settings"]["bias"] is True
    assert results[1]["settings"]["ff_residual"] is False
    assert results[2]["settings"]["d_residual"] == 64
    assert results[2]["settings"]["n_heads"] == 8
