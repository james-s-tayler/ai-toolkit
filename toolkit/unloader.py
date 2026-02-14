import gc
import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )
        return None

    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    def to(self, *args, **kwargs):
        return self


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            # For models with text encoder as a list
            # Store references to all encoders before clearing the list
            encoders_to_unload = []
            for encoder in model.text_encoder:
                # Keep track of real encoders (not fakes) that need unloading
                if not isinstance(encoder, FakeTextEncoder):
                    encoders_to_unload.append(encoder)
            
            # Clear the list first to remove all references
            model.text_encoder.clear()
            
            # Now unload the stored encoders
            for encoder in encoders_to_unload:
                # Move to CPU
                encoder.to('cpu')
                # Clear all parameters and buffers
                for param in encoder.parameters():
                    param.data = None
                # Delete the encoder
                del encoder
            
            # Clear the temporary list
            encoders_to_unload.clear()
            del encoders_to_unload
            
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                # Store reference to old encoder before replacing
                text_encoder_to_unload = pipe.text_encoder
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                # Replace with fake encoder first to remove pipeline reference
                pipe.text_encoder = te
                # Now move old encoder to CPU to free GPU memory (if not already fake)
                if not isinstance(text_encoder_to_unload, FakeTextEncoder):
                    # Move to CPU
                    text_encoder_to_unload.to('cpu')
                    # Clear all parameters
                    for param in text_encoder_to_unload.parameters():
                        param.data = None
                    # Explicitly delete the old text encoder to free system RAM
                    del text_encoder_to_unload
                    text_encoder_to_unload = None

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                # Store reference to old encoder before replacing
                text_encoder_to_unload = getattr(pipe, f"text_encoder_{i}")
                if text_encoder_to_unload is not None:
                    te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                    text_encoder_list.append(te)
                    # Replace with fake encoder first to remove pipeline reference
                    setattr(pipe, f"text_encoder_{i}", te)
                    # Now move old encoder to CPU to free GPU memory (if not already fake)
                    if not isinstance(text_encoder_to_unload, FakeTextEncoder):
                        # Move to CPU
                        text_encoder_to_unload.to('cpu')
                        # Clear all parameters
                        for param in text_encoder_to_unload.parameters():
                            param.data = None
                        # Explicitly delete the old text encoder to free system RAM
                        del text_encoder_to_unload
                        text_encoder_to_unload = None
                i += 1
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder
            # Store reference to old encoder before replacing
            text_encoder_to_unload = model.text_encoder
            # Replace with fake encoder first to remove model reference
            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            # Now move old encoder to CPU to free GPU memory
            # Move to CPU
            text_encoder_to_unload.to('cpu')
            # Clear all parameters
            for param in text_encoder_to_unload.parameters():
                param.data = None
            # Explicitly delete the old text encoder to free system RAM
            del text_encoder_to_unload
            text_encoder_to_unload = None

    # Also clear tokenizer references if they exist (can be large)
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        if isinstance(model.tokenizer, list):
            for tok in model.tokenizer:
                del tok
            model.tokenizer.clear()
            # Replace with minimal placeholder
            model.tokenizer = [None]
        else:
            del model.tokenizer
            model.tokenizer = None
    
    # Clear any cached text-related components from pipeline
    if hasattr(model, 'pipeline') and model.pipeline is not None:
        pipe = model.pipeline
        # Clear tokenizer from pipeline
        if hasattr(pipe, 'tokenizer'):
            pipe.tokenizer = None
        # Clear connectors (text-related) from pipeline  
        if hasattr(pipe, 'connectors'):
            if pipe.connectors is not None:
                pipe.connectors.to('cpu')
                del pipe.connectors
            pipe.connectors = None

    # Aggressively free memory
    flush()
    # Force full garbage collection including all generations
    # Multiple passes help clean up circular references in large model graphs
    gc.collect(2)  # Collect generation 2 (oldest objects)
    gc.collect(2)  # Second pass to catch any newly unreachable objects
    gc.collect(2)  # Third pass for good measure
    torch.cuda.empty_cache()
    # Final sync to ensure GPU operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
