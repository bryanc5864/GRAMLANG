#!/usr/bin/env python
"""
Download and verify all 9 pretrained genomic models.
Downloads sequentially to manage disk space.
Tests each model with a dummy sequence.
"""

import os
import sys
import gc
import json
import torch
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = {}
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def get_model_size_mb(model):
    """Estimate model size in MB from parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 ** 2)  # FP32 bytes


def test_model(name, load_fn, test_fn):
    """Load, test, and report on a model."""
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    try:
        model, tokenizer = load_fn()
        size_mb = get_model_size_mb(model)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Size: {size_mb:.1f} MB (FP32)")

        # Test with dummy sequence
        result = test_fn(model, tokenizer)
        print(f"  Test output shape/value: {result}")

        RESULTS[name] = {
            'status': 'OK',
            'params': sum(p.numel() for p in model.parameters()),
            'size_mb': size_mb,
            'test_result': str(result),
        }

        # Clean up
        del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  Status: OK")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        RESULTS[name] = {
            'status': 'FAILED',
            'error': str(e),
        }
        gc.collect()
        torch.cuda.empty_cache()


def load_dnabert2():
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    name = "zhihan1996/DNABERT-2-117M"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    ModelClass = get_class_from_dynamic_module(
        'bert_layers.BertModel', name, code_revision=None
    )
    mod = ModelClass.from_pretrained(name).cuda().eval()
    return mod, tok

def test_dnabert2(model, tokenizer):
    seq = "ACGTACGTACGT" * 10
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    # Custom model returns tuple: (last_hidden_state, pooled_output)
    return f"last_hidden_state shape={out[0].shape}, pooled shape={out[1].shape}"


def load_nt():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mod = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True).cuda().eval()
    return mod, tok

def test_nt(model, tokenizer):
    seq = "ACGTACGTACGT" * 10
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return f"hidden_states layers={len(out.hidden_states)}, shape={out.hidden_states[-1].shape}"


def load_caduceus():
    from transformers import AutoModel, AutoTokenizer
    name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mod = AutoModel.from_pretrained(name, trust_remote_code=True).cuda().eval()
    return mod, tok

def test_caduceus(model, tokenizer):
    seq = "ACGTACGTACGT" * 10
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return f"hidden_states layers={len(out.hidden_states)}, shape={out.hidden_states[-1].shape}"


def load_hyenadna():
    from transformers import AutoModel, AutoTokenizer
    name = "LongSafari/hyenadna-large-1m-seqlen-hf"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mod = AutoModel.from_pretrained(name, trust_remote_code=True).cuda().eval()
    return mod, tok

def test_hyenadna(model, tokenizer):
    seq = "ACGTACGTACGT" * 10
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return f"hidden_states layers={len(out.hidden_states)}, shape={out.hidden_states[-1].shape}"


def load_enformer():
    from enformer_pytorch import Enformer
    mod = Enformer.from_pretrained('EleutherAI/enformer-official-rough').cuda().eval()
    return mod, None

def test_enformer(model, tokenizer):
    import numpy as np
    # Create minimal input
    seq_len = 196608
    # Use random one-hot
    np.random.seed(42)
    bases = np.eye(4)[np.random.randint(0, 4, seq_len)]
    tensor = torch.tensor(bases, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(tensor)
    return f"human shape={out['human'].shape}"


if __name__ == '__main__':
    print("=" * 60)
    print("GRAMLANG Model Download & Verification")
    print("=" * 60)

    # Test models from smallest to largest
    models_to_test = [
        ("DNABERT-2", load_dnabert2, test_dnabert2),
        ("Nucleotide Transformer v2-500M", load_nt, test_nt),
        ("Caduceus", load_caduceus, test_caduceus),
        ("HyenaDNA", load_hyenadna, test_hyenadna),
        ("Enformer", load_enformer, test_enformer),
    ]

    for name, load_fn, test_fn in models_to_test:
        test_model(name, load_fn, test_fn)

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'model_verification.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in RESULTS.items():
        status = result['status']
        if status == 'OK':
            print(f"  {name}: OK ({result.get('size_mb', 0):.0f} MB)")
        else:
            print(f"  {name}: FAILED - {result.get('error', 'unknown')}")

    print(f"\nResults saved to: {output_path}")
