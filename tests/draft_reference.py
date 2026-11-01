"""Independent FP32 NumPy oracle for the pinned BF16 DFlash2 synthetic fixture.

This is a development correctness check, not an experiment evaluation. Run after
`draft-self-test --output-dir ...`; never run concurrently with GPU timing.
Requires numpy. It reads public weights locally and writes no weights or prompts.
Reference architecture: z-lab/dflash DFlash2DraftModel (model.py).
"""
import argparse
import json
import pathlib
import struct

import numpy as np


class Tensors:
    def __init__(self, path):
        self.path = path
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.offset = n + 8

    def get(self, name):
        t = self.header[name]
        assert t["dtype"] == "BF16", name
        raw = np.memmap(self.path, dtype="<u2", mode="r", offset=self.offset + t["data_offsets"][0], shape=tuple(t["shape"]))
        return (raw.astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fixture-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    draft = Tensors(args.draft_dir / "model.safetensors")
    target = Tensors(args.model_dir / "outside.safetensors")
    fixture = json.loads((args.fixture_dir / "draft-fixture.json").read_text())
    context, width, anchor = (fixture[k] for k in ("context", "width", "anchor"))

    def project(name, x):
        output = x @ draft.get(name + ".weight").T
        assert np.all(np.isfinite(output)), name
        return output

    def norm(name, x):
        return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6))) * draft.get(name + ".weight")

    def conv(name, x, dynamic, side):
        base = draft.get(name + ".base_kernel")[side]
        shaped = x.reshape(width, 128, 16)
        out = np.zeros_like(shaped)
        for offset in range(2):
            values = shaped if offset == 0 else np.concatenate([np.zeros_like(shaped[:1]), shaped[:-1]], axis=0)
            out += base[offset].reshape(128, 16) * values
            out += dynamic[:, side, offset, :, None] * values
        return out.reshape(width, 2048)

    def prepare(name, x):
        dynamic = project(name + ".kernel_projection", x).reshape(width, 2, 2, 128)
        return conv(name, x, dynamic, 0), dynamic

    def rope(x, positions):
        angle = positions.astype(np.float32)[:, None, None] * np.power(np.float32(1e7), -np.arange(64, dtype=np.float32)[None, None, :] / np.float32(64))
        cos, sin = np.cos(angle), np.sin(angle)
        return np.concatenate([x[..., :64] * cos - x[..., 64:] * sin, x[..., 64:] * cos + x[..., :64] * sin], axis=-1)

    features = (((np.arange(context * 16384, dtype=np.int64) * 13) % 2001 - 1000).astype(np.float32) * np.float32(0.001)).reshape(context, 16384)
    ctx = norm("hidden_norm", project("fc", features))
    token_ids = [anchor] + [248077] * (width - 1)
    embeddings = target.get("model.language_model.embed_tokens.weight")
    hidden = embeddings[token_ids].copy()
    del embeddings
    for layer in range(6):
        path = f"layers.{layer}"
        prepared, dynamic = prepare(path + ".attention_conv", norm(path + ".input_layernorm", hidden))
        q = norm(path + ".self_attn.q_norm", project(path + ".self_attn.q_proj", prepared).reshape(width, 32, 128))
        k = np.concatenate([project(path + ".self_attn.k_proj", ctx), project(path + ".self_attn.k_proj", prepared)]).reshape(context + width, 8, 128)
        k = norm(path + ".self_attn.k_norm", k)
        v = np.concatenate([project(path + ".self_attn.v_proj", ctx), project(path + ".self_attn.v_proj", prepared)]).reshape(context + width, 8, 128)
        q = rope(q, np.arange(context, context + width))
        k = np.repeat(rope(k, np.arange(context + width)), 4, axis=1)
        v = np.repeat(v, 4, axis=1)
        scores = np.einsum("qhd,khd->qhk", q, k, optimize=True) / np.sqrt(np.float32(128))
        visible = (np.arange(context, context + width)[:, None] - np.arange(context + width)[None, :]) < 2048
        scores = np.where(visible[:, None, :], scores, -np.inf)
        probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        attention = np.einsum("qhk,khd->qhd", probabilities, v, optimize=True).reshape(width, 4096)
        hidden += conv(path + ".attention_conv", project(path + ".self_attn.o_proj", attention), dynamic, 1)
        prepared, dynamic = prepare(path + ".mlp_conv", norm(path + ".post_attention_layernorm", hidden))
        gate = project(path + ".mlp.gate_proj", prepared)
        activation = (gate / (1 + np.exp(-gate))) * project(path + ".mlp.up_proj", prepared)
        hidden += conv(path + ".mlp_conv", project(path + ".mlp.down_proj", activation), dynamic, 1)
    hidden = norm("norm", hidden)
    logits = hidden[1:] @ target.get("lm_head.weight").T
    codes = project("candidate_selector.hidden_projection", hidden[1:])
    pre, post = draft.get("candidate_selector.predecessor_codebook"), draft.get("candidate_selector.successor_codebook")
    predecessor = anchor
    proposals = []
    for row in range(width - 1):
        candidates = np.argpartition(logits[row], -16)[-16:]
        scores = logits[row, candidates] + (pre[predecessor] * codes[row]) @ post[candidates].T
        predecessor = int(candidates[np.argmax(scores)])
        proposals.append(predecessor)
    actual_context = np.fromfile(args.fixture_dir / "draft-context.f32", dtype="<f4").reshape(context, 2048)
    actual_hidden = np.fromfile(args.fixture_dir / "draft-hidden.f32", dtype="<f4").reshape(width, 2048)
    report = {"oracle": "independent NumPy FP32", "context_max_abs_error": float(np.max(np.abs(ctx - actual_context))), "hidden_max_abs_error": float(np.max(np.abs(hidden - actual_hidden))), "hidden_rms_error": float(np.sqrt(np.mean((hidden - actual_hidden) ** 2))), "expected_proposals": proposals, "actual_proposals": fixture["proposals"]}
    report["passed"] = bool(np.allclose(ctx, actual_context, atol=1e-3, rtol=1e-3) and np.allclose(hidden, actual_hidden, atol=1e-3, rtol=1e-3) and proposals == fixture["proposals"])
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
