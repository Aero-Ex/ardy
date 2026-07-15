# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for ARDY text conditioning."""

import gc
import platform
import os

import numpy as np
import torch
from torch import nn

from .llm2vec import LLM2Vec


class LLM2VecEncoder(nn.Module):
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.llm_dim = llm_dim

        env_device = os.environ.get("TEXT_ENCODER_DEVICE")
        if env_device:
            device = env_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        custom_path = os.path.expanduser("~/llm2vec-model")
        if os.path.exists(custom_path):
            self.custom_dir = custom_path
        else:
                root_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir))
                self.custom_dir = os.path.abspath(os.path.join(root_path, "models", "KIMODO-Meta3_llm2vec_NF4"))
                if not os.path.exists(self.custom_dir):
                    # Fall back to Hugging Face repo ID
                    self.custom_dir = "Aero-Ex/KIMODO-Meta3_llm2vec_NF4"

        print(f"[LLM2VecEncoder] Initializing model from {self.custom_dir}...")
        print(f"[LLM2VecEncoder] Initialized (Waiting for first use to load weights)...")
        self.model = None

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.torch_dtype = dtype
        if device is not None:
            self._device = str(device) if not isinstance(device, str) else device
        if self.model is not None:
            self.model.to(device=device, dtype=dtype)
        return self

    def unload(self):
        """Offload the model weights to System RAM (CPU) if currently on GPU."""
        if self.model is not None:
            if self.get_device().type == "cuda":
                print(f"[LLM2VecEncoder] Offloading 5.4GB model to System RAM...")
                self.model.model.to("cpu")
                gc.collect()
                if platform.system() == "Linux":
                    try:
                        import ctypes
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass
                elif platform.system() == "Windows":
                    from ardy.model.memory_manager import release_system_memory
                    release_system_memory()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    def reload(self):
        """Move from System RAM to VRAM."""
        if self.model is None:
            self.model = LLM2Vec.from_pretrained(
                base_model_name_or_path=self.custom_dir,
                peft_model_name_or_path=None,
                torch_dtype=self.torch_dtype,
                device_map="cpu"
            )
            self.model.model.config._name_or_path = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

        from ardy.model.memory_manager import manager
        # Need ~5.4GB of VRAM
        manager.ensure_vram_capacity(5400 * 1024 * 1024, device=self._device, exclude_name="text_encoder")

        curr_device = self.get_device()
        if curr_device.type != "cuda":
            if torch.backends.mps.is_available():
                print(f"[LLM2VecEncoder] Moving weights to GPU (mps)...")
                self.model.model.to("mps")
            else:
                print(f"[LLM2VecEncoder] Moving weights to GPU ({self._device})...")
                self.model.model.to(self._device)
            
            gc.collect()
            
            if platform.system() == "Linux":
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            elif platform.system() == "Windows":
                from ardy.model.memory_manager import release_system_memory
                release_system_memory()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            manager.log_memory_usage("Encoder Transfer Complete (RAM Reclaimed)")
        else:
            print(f"[LLM2VecEncoder] Model already on GPU ({curr_device})")

    def get_device(self):
        if self.model is None:
            return torch.device("cpu")
        for p in self.model.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def delete(self):
        """Reclaim RAM without deleting from disk unless absolutely necessary."""
        self.unload()

    def __call__(self, text: list[str] | str):
        self.reload() # Auto-reload if called
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        results = []
        for t in text:
            with torch.no_grad():
                emb = self.model.encode([t])
                results.append(emb)

        encoded_text = np.concatenate(results, axis=0)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())

        # Automatically unload the model from GPU to free VRAM for diffusion/denoising
        from ardy.model.memory_manager import manager
        if manager.offload_enabled:
            print("[LLM2VecEncoder] Auto-offloading after encode call...")
            self.unload()

        return encoded_text, lengths
