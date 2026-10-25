#!/usr/bin/env python3
"""Torch-free port of whisper.cpp/models/convert-h5-to-ggml.py.

Reads a Hugging Face Whisper checkpoint (safetensors) with numpy only and writes
the ggml binary whisper.cpp expects. Avoids downloading torch/transformers over a
slow link. Output tensors are f16 except 1-D tensors, conv biases and positional
embeddings, which stay f32 (same rule as the upstream script).

Usage: convert_whisper_hf_to_ggml.py <hf_model_dir> <openai_whisper_repo_dir> <out_dir>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

CONV_MAP = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "proj_out.weight": "decoder.proj.weight",
}


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    dir_model, dir_whisper, dir_out = (Path(p) for p in sys.argv[1:4])
    hparams = json.load((dir_model / "config.json").open(encoding="utf8"))
    if not isinstance(hparams.get("max_length"), int):
        hparams["max_length"] = hparams.get("max_target_positions", 448)
    tokens = json.load((dir_model / "vocab.json").open(encoding="utf8"))

    n_mels = hparams["num_mel_bins"]
    with np.load(dir_whisper / "whisper" / "assets" / "mel_filters.npz") as f:
        filters = f[f"mel_{n_mels}"].astype(np.float32)

    weights = {}
    for st in sorted(dir_model.glob("*.safetensors")):
        weights.update(load_file(str(st)))
    print(f"loaded {len(weights)} tensors from {dir_model}")

    dir_out.mkdir(parents=True, exist_ok=True)
    fname_out = dir_out / "ggml-model.bin"
    with open(fname_out, "wb") as fout:
        fout.write(struct.pack("i", 0x67676D6C))
        for k in ("vocab_size", "max_source_positions", "d_model", "encoder_attention_heads", "encoder_layers",
                  "max_length", "d_model", "decoder_attention_heads", "decoder_layers", "num_mel_bins"):
            fout.write(struct.pack("i", hparams[k]))
        fout.write(struct.pack("i", 1))  # use_f16
        fout.write(struct.pack("i", filters.shape[0]))
        fout.write(struct.pack("i", filters.shape[1]))
        filters.astype("<f4").tofile(fout)

        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        fout.write(struct.pack("i", len(tokens)))
        for text_tok, _ in sorted(tokens.items(), key=lambda x: x[1]):
            b = bytearray(byte_decoder[c] for c in text_tok)
            fout.write(struct.pack("i", len(b)))
            fout.write(b)

        for src, arr in weights.items():
            if src == "proj_out.weight":
                print("Skipping", src)
                continue
            nn = src.split(".")[1:] if src != "proj_out.weight" else src.split(".")
            if nn[1] == "layers":
                nn[1] = "blocks"
                inner = ".".join(nn[3:-1])
                if inner == "encoder_attn.k_proj":
                    mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
                else:
                    mapped = CONV_MAP[inner]
                name = ".".join(nn[:3] + [mapped] + nn[-1:])
            else:
                name = ".".join(nn)
                name = CONV_MAP.get(name, name)

            data = np.asarray(arr).squeeze().astype(np.float16)
            if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(data.shape[0], 1)
            n_dims = data.ndim
            ftype = 1
            if n_dims < 2 or name in ("encoder.conv1.bias", "encoder.conv2.bias",
                                      "encoder.positional_embedding", "decoder.positional_embedding"):
                data = data.astype(np.float32)
                ftype = 0
            s = name.encode("utf-8")
            fout.write(struct.pack("iii", n_dims, len(s), ftype))
            for i in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            fout.write(s)
            data.tofile(fout)
    print("Done. Output file:", fname_out)


if __name__ == "__main__":
    main()
