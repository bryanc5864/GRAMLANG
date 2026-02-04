"""
Unified model loading interface for GRAMLANG.

Provides a consistent GrammarModel interface for all 9 pretrained models.
Models are loaded one at a time to stay within memory budget.
"""

import os
import gc
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.utils.sequence import one_hot_encode, pad_sequence, reverse_complement


class GrammarModel(ABC):
    """Unified interface for all 9 genomic models."""

    @abstractmethod
    def predict_expression(self, sequences: List[str],
                           cell_type: str = None) -> np.ndarray:
        """Predict expression for a batch of sequences."""
        pass

    @abstractmethod
    def get_embeddings(self, sequences: List[str],
                       layer: int = -1) -> np.ndarray:
        """Extract pooled embeddings from a specific layer."""
        pass

    @abstractmethod
    def get_all_layer_embeddings(self, sequences: List[str]) -> List[np.ndarray]:
        """Extract embeddings from ALL layers."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def architecture_type(self) -> str:
        """One of: 'transformer', 'cnn', 'ssm', 'hybrid'"""
        pass

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        pass

    def _batch_process(self, sequences: List[str], fn, batch_size: int = 8):
        """Process sequences in batches to manage memory."""
        results = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            results.append(fn(batch))
        return np.concatenate(results, axis=0)

    def unload(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EnformerGrammarModel(GrammarModel):
    """Enformer wrapper."""

    CAGE_TRACKS = {
        'K562': [4828, 4829, 4830, 4831],
        'HepG2': [4710, 4711, 4712, 4713],
        'GM12878': [4654, 4655, 4656, 4657],
    }

    def __init__(self, device='cuda'):
        from enformer_pytorch import Enformer
        self.device = device
        self.model = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        self.model.eval()
        self.model.to(device)
        self._hooks_data = {}

    def predict_expression(self, sequences, cell_type='K562'):
        tracks = self.CAGE_TRACKS.get(cell_type, self.CAGE_TRACKS['K562'])

        def _predict_batch(batch):
            preds = []
            for seq in batch:
                padded = pad_sequence(seq, 196608)
                encoded = one_hot_encode(padded)
                tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(tensor)
                    human_pred = output['human']
                    center = human_pred.shape[1] // 2
                    expr = human_pred[0, center, tracks].mean().cpu().item()
                    preds.append(expr)
            return np.array(preds)

        return self._batch_process(sequences, _predict_batch, batch_size=4)

    def get_embeddings(self, sequences, layer=-1):
        embeddings = []
        transformer_layers = list(self.model.transformer)
        for seq in sequences:
            padded = pad_sequence(seq, 196608)
            encoded = one_hot_encode(padded)
            tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)

            captured = {}
            def hook_fn(module, inp, out):
                captured['output'] = out.detach().cpu()

            target = transformer_layers[layer]
            handle = target.register_forward_hook(hook_fn)
            with torch.no_grad():
                self.model(tensor)
            handle.remove()

            emb = captured['output'].numpy().mean(axis=1).squeeze()
            embeddings.append(emb)
        return np.array(embeddings)

    def get_all_layer_embeddings(self, sequences):
        transformer_layers = list(self.model.transformer)
        layers_embs = [[] for _ in range(self.num_layers)]
        for seq in sequences:
            padded = pad_sequence(seq, 196608)
            encoded = one_hot_encode(padded)
            tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)

            captures = {}
            handles = []
            for li, layer_mod in enumerate(transformer_layers):
                def make_hook(idx):
                    def hook_fn(m, i, o):
                        captures[idx] = o.detach().cpu()
                    return hook_fn
                handles.append(layer_mod.register_forward_hook(make_hook(li)))

            with torch.no_grad():
                self.model(tensor)

            for h in handles:
                h.remove()

            for li in range(self.num_layers):
                emb = captures[li].numpy().mean(axis=1).squeeze()
                layers_embs[li].append(emb)

        return [np.array(le) for le in layers_embs]

    @property
    def name(self): return "enformer"
    @property
    def architecture_type(self): return "transformer"
    @property
    def hidden_dim(self): return 1536
    @property
    def num_layers(self): return 11


class NTGrammarModel(GrammarModel):
    """Nucleotide Transformer v2-500M wrapper."""

    def __init__(self, device='cuda'):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.device = device
        model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        self.model.eval()
        self._probe = None  # Set after training

    def set_probe(self, probe):
        """Set the expression prediction probe."""
        self._probe = probe

    def predict_expression(self, sequences, cell_type=None):
        if self._probe is None:
            raise RuntimeError("Expression probe not trained. Call set_probe() first.")
        embs = self.get_embeddings(sequences)
        embs_tensor = torch.tensor(embs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self._probe(embs_tensor).cpu().numpy()
        return preds

    def get_embeddings(self, sequences, layer=-1):
        def _get_batch(batch):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1000).to(self.device)
            with torch.no_grad():
                out = self.model(**tokens, output_hidden_states=True)
                hidden = out.hidden_states[layer]  # (batch, seq, dim)
                pooled = hidden.mean(dim=1).cpu().numpy()
            return pooled
        return self._batch_process(sequences, _get_batch, batch_size=16)

    def get_all_layer_embeddings(self, sequences):
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=True, max_length=1000).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens, output_hidden_states=True)
        return [h.mean(dim=1).cpu().numpy() for h in out.hidden_states]

    @property
    def name(self): return "nt"
    @property
    def architecture_type(self): return "transformer"
    @property
    def hidden_dim(self): return 1024
    @property
    def num_layers(self): return 25


class DNABERT2GrammarModel(GrammarModel):
    """DNABERT-2 wrapper. Uses custom model class to bypass AutoModel config mismatch."""

    def __init__(self, device='cuda'):
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        self.device = device
        model_name = "zhihan1996/DNABERT-2-117M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        ModelClass = get_class_from_dynamic_module(
            'bert_layers.BertModel', model_name, code_revision=None
        )
        self.model = ModelClass.from_pretrained(model_name).to(device)
        self.model.eval()
        self._probe = None

    def set_probe(self, probe):
        self._probe = probe

    def predict_expression(self, sequences, cell_type=None):
        if self._probe is None:
            raise RuntimeError("Expression probe not trained.")
        embs = self.get_embeddings(sequences)
        embs_tensor = torch.tensor(embs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self._probe(embs_tensor).cpu().numpy()
        return preds

    def get_embeddings(self, sequences, layer=-1):
        def _get_batch(batch):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.model(**tokens)
                # Custom DNABERT-2 returns tuple: (last_hidden_state, pooled_output)
                last_hidden = out[0]  # (batch, seq, 768)
                pooled = last_hidden.mean(dim=1).cpu().numpy()
            return pooled
        return self._batch_process(sequences, _get_batch, batch_size=32)

    def get_all_layer_embeddings(self, sequences):
        # DNABERT-2 custom model doesn't support output_hidden_states in tuple mode
        # Use forward hooks to capture all layer outputs
        # Note: DNABERT-2 uses unpadded inputs, so layer outputs are 2D (total_nnz, hidden_dim)
        captures = {}
        handles = []
        for li, layer_mod in enumerate(self.model.encoder.layer):
            def make_hook(idx):
                def hook_fn(m, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    captures[idx] = t.detach().cpu()
                return hook_fn
            handles.append(layer_mod.register_forward_hook(make_hook(li)))

        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            self.model(**tokens)

        for h in handles:
            h.remove()

        result = []
        for i in range(len(captures)):
            t = captures[i]
            # Handle both 2D (unpadded: total_nnz, dim) and 3D (batch, seq, dim)
            if t.dim() == 2:
                result.append(t.mean(dim=0).unsqueeze(0).numpy())
            else:
                result.append(t.mean(dim=1).numpy())
        return result

    @property
    def name(self): return "dnabert2"
    @property
    def architecture_type(self): return "transformer"
    @property
    def hidden_dim(self): return 768
    @property
    def num_layers(self): return 12


class HyenaDNAGrammarModel(GrammarModel):
    """HyenaDNA wrapper."""

    def __init__(self, device='cuda'):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        model_name = "LongSafari/hyenadna-large-1m-seqlen-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self._probe = None

    def set_probe(self, probe):
        self._probe = probe

    def predict_expression(self, sequences, cell_type=None):
        if self._probe is None:
            raise RuntimeError("Expression probe not trained.")
        embs = self.get_embeddings(sequences)
        embs_tensor = torch.tensor(embs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self._probe(embs_tensor).cpu().numpy()
        return preds

    def get_embeddings(self, sequences, layer=-1):
        def _get_batch(batch):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                out = self.model(**tokens, output_hidden_states=True)
                hidden = out.hidden_states[layer]
                pooled = hidden.mean(dim=1).cpu().numpy()
            return pooled
        return self._batch_process(sequences, _get_batch, batch_size=16)

    def get_all_layer_embeddings(self, sequences):
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens, output_hidden_states=True)
        return [h.mean(dim=1).cpu().numpy() for h in out.hidden_states]

    @property
    def name(self): return "hyenadna"
    @property
    def architecture_type(self): return "ssm"
    @property
    def hidden_dim(self): return 256
    @property
    def num_layers(self): return 10


class CaduceusGrammarModel(GrammarModel):
    """Caduceus (BiMamba) wrapper."""

    def __init__(self, device='cuda'):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self._probe = None

    def set_probe(self, probe):
        self._probe = probe

    def predict_expression(self, sequences, cell_type=None):
        if self._probe is None:
            raise RuntimeError("Expression probe not trained.")
        embs = self.get_embeddings(sequences)
        embs_tensor = torch.tensor(embs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self._probe(embs_tensor).cpu().numpy()
        return preds

    def get_embeddings(self, sequences, layer=-1):
        def _get_batch(batch):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                out = self.model(**tokens, output_hidden_states=True)
                hidden = out.hidden_states[layer]
                pooled = hidden.mean(dim=1).cpu().numpy()
            return pooled
        return self._batch_process(sequences, _get_batch, batch_size=32)

    def get_all_layer_embeddings(self, sequences):
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens, output_hidden_states=True)
        return [h.mean(dim=1).cpu().numpy() for h in out.hidden_states]

    @property
    def name(self): return "caduceus"
    @property
    def architecture_type(self): return "ssm"
    @property
    def hidden_dim(self): return 256
    @property
    def num_layers(self): return 16


class GPNGrammarModel(GrammarModel):
    """GPN-MSA wrapper."""

    def __init__(self, device='cuda'):
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        model_name = "songlab/gpn-msa-sapiens"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self._probe = None

    def set_probe(self, probe):
        self._probe = probe

    def predict_expression(self, sequences, cell_type=None):
        if self._probe is None:
            raise RuntimeError("Expression probe not trained.")
        embs = self.get_embeddings(sequences)
        embs_tensor = torch.tensor(embs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self._probe(embs_tensor).cpu().numpy()
        return preds

    def get_embeddings(self, sequences, layer=-1):
        def _get_batch(batch):
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.model(**tokens, output_hidden_states=True)
                hidden = out.hidden_states[layer]
                pooled = hidden.mean(dim=1).cpu().numpy()
            return pooled
        return self._batch_process(sequences, _get_batch, batch_size=32)

    def get_all_layer_embeddings(self, sequences):
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens, output_hidden_states=True)
        return [h.mean(dim=1).cpu().numpy() for h in out.hidden_states]

    @property
    def name(self): return "gpn"
    @property
    def architecture_type(self): return "cnn"
    @property
    def hidden_dim(self): return 512
    @property
    def num_layers(self): return 25


# Deferred model classes (require special handling)
# Borzoi, Sei, and Evo are more complex to load and will be implemented
# as needed based on their specific package requirements.


def load_model(model_name: str, device: str = 'cuda',
               probe_dir: str = None, dataset_name: str = 'vaishnav2022') -> GrammarModel:
    """
    Load a single model by name, auto-loading expression probe if available.

    Args:
        model_name: One of 'enformer', 'nt', 'dnabert2', 'hyenadna',
                    'caduceus', 'gpn', 'borzoi', 'sei', 'evo'
        device: CUDA device string
        probe_dir: Directory containing trained probes (auto-detected if None)
        dataset_name: Dataset the probe was trained on

    Returns:
        GrammarModel instance (with probe loaded if available)
    """
    loaders = {
        'enformer': EnformerGrammarModel,
        'nt': NTGrammarModel,
        'dnabert2': DNABERT2GrammarModel,
        'hyenadna': HyenaDNAGrammarModel,
        'caduceus': CaduceusGrammarModel,
        'gpn': GPNGrammarModel,
    }

    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(loaders.keys())}")

    print(f"Loading model: {model_name}...")
    model = loaders[model_name](device=device)
    print(f"  Loaded {model_name} ({model.architecture_type}, "
          f"hidden_dim={model.hidden_dim}, layers={model.num_layers})")

    # Auto-load expression probe for foundation models
    if hasattr(model, 'set_probe') and model._probe is None:
        if probe_dir is None:
            # Auto-detect probe directory
            import os
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            probe_dir = os.path.join(project_dir, 'data', 'probes')

        probe_name = f'{model_name}_{dataset_name}'
        probe_path = os.path.join(probe_dir, f'{probe_name}_probe.pt')
        if os.path.exists(probe_path):
            from src.models.expression_probes import load_probe
            probe = load_probe(probe_dir, probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            print(f"  Loaded expression probe from {probe_path}")
        else:
            print(f"  WARNING: No expression probe found at {probe_path}")
            print(f"  predict_expression() will fail. Run train_probes.py first.")

    return model


def load_models_sequential(model_names: List[str], device: str = 'cuda') -> Dict[str, GrammarModel]:
    """
    Load multiple models sequentially (one at a time, unloading previous).
    Use this when running experiments across models to manage memory.
    """
    models = {}
    for name in model_names:
        models[name] = load_model(name, device)
    return models
