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
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder"):
                # Store reference to old encoder to delete after replacement
                old_te = pipe.text_encoder
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                # Move to CPU before deleting to free GPU memory first
                old_te.to('cpu')
                pipe.text_encoder = te
                # Explicitly delete the old text encoder to free system RAM
                del old_te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                # Store reference to old encoder to delete after replacement
                old_te = getattr(pipe, f"text_encoder_{i}")
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                # Move to CPU before deleting to free GPU memory first
                old_te.to('cpu')
                setattr(pipe, f"text_encoder_{i}", te)
                # Explicitly delete the old text encoder to free system RAM
                del old_te
                i += 1
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder
            # Store reference to old encoder to delete after replacement
            old_te = model.text_encoder
            # Move to CPU before deleting to free GPU memory first
            old_te.to('cpu')
            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            # Explicitly delete the old text encoder to free system RAM
            del old_te

    flush()
