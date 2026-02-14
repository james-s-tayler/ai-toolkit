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
            old_text_encoder_list = model.text_encoder
            text_encoder_list = []

            # Build new list and update pipeline
            for i, te in enumerate(old_text_encoder_list):
                # Create fake encoder
                te_fake = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te_fake)
                
                # Determine pipeline attribute name
                # text_encoder (i=0), text_encoder_2 (i=1), text_encoder_3 (i=2), etc.
                te_attr = "text_encoder" if i == 0 else f"text_encoder_{i+1}"
                if hasattr(model, 'pipeline') and model.pipeline is not None and hasattr(model.pipeline, te_attr):
                    setattr(model.pipeline, te_attr, te_fake)
            
            # Replace list atomically
            model.text_encoder = text_encoder_list
            
            # Now cleanup old encoders
            for te in old_text_encoder_list:
                te.to('cpu')
            
            # Delete the old list to free references
            del old_text_encoder_list
                
        else:
            # Single text encoder
            old_te = model.text_encoder
            te_fake = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            
            # Replace in model and pipeline
            model.text_encoder = te_fake
            if hasattr(model, 'pipeline') and model.pipeline is not None and hasattr(model.pipeline, 'text_encoder'):
                model.pipeline.text_encoder = te_fake
            
            # Cleanup old encoder
            old_te.to('cpu')

    flush()
