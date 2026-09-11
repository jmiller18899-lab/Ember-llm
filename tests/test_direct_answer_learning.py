import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest
import torch
import torch.nn.functional as F

from direct_answers.learn import batch, configure_training, encode_row, frozen_hash, validate_data


def small_model(tmp_path):
    with zipfile.ZipFile("ember-v0.0.7-hf-ready.zip") as z:
        source = z.read("ember/src/model.py")
    path = tmp_path / "model.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("canary_test_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EmberGPT(module.ModelConfig(vocab_size=64, block_size=32, n_embd=16, n_head=2, n_layer=6))


def test_real_optimizer_step_preserves_all_routing_parameters(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = small_model(tmp_path).eval()
    selected = configure_training(model)
    before = frozen_hash(model)
    old_trainable = {k: v.detach().clone() for k, v in selected.items()}
    inputs = torch.randint(0, 64, (2, 10))
    captured = []
    handle = model.blocks[4].register_forward_hook(lambda _m, _i, out: captured.append(out.detach().clone()))
    model(inputs)
    optimizer = torch.optim.AdamW(selected.values(), lr=0.01)
    x, y = batch([(inputs[0], inputs[0].roll(-1)), (inputs[1, :6], inputs[1, :6].roll(-1))])
    logits, _ = model(x)
    F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100).backward()
    optimizer.step()
    model(inputs)
    handle.remove()
    assert frozen_hash(model) == before
    torch.testing.assert_close(captured[0], captured[-1], rtol=0, atol=0)
    assert any(not torch.equal(old_trainable[k], v) for k, v in selected.items())
    assert not model.token_emb.weight.requires_grad
    assert not model.lm_head.weight.requires_grad


def test_loss_masks_prompt_and_padding():
    class Tokenizer:
        def encode(self, text):
            return list(text.encode())
    x, y = encode_row(Tokenizer(), {"user": "hello", "answer": "Hello!"}, 2048)
    assert (y == -100).any()
    assert (y != -100).sum() == len("Hello!<|endoftext|>")
    _, padded = batch([(x, y), (x[:-3], y[:-3])])
    assert (padded[1, -3:] == -100).all()


def test_boundary_mismatch_stops_before_training():
    class Tokenizer:
        def encode(self, text):
            return [len(text)]
    with pytest.raises(RuntimeError, match="boundary"):
        encode_row(Tokenizer(), {"user": "hello", "answer": "Hello!"}, 2048)


def test_data_splits_exclude_router_and_old_quality_requests():
    data = json.loads(Path("direct_answers/canary-data.json").read_text())
    router = json.loads(Path("tool_assistant/data/router-training.json").read_text())["cases"]
    assert validate_data(data, router)
    data["confirmation"][0]["user"] = data["train"][0]["user"]
    with pytest.raises(ValueError, match="leakage"):
        validate_data(data, router)
