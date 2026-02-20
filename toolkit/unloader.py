import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING
from toolkit.print import print_verbose


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
    verbose = model.model_config.verbose if hasattr(model.model_config, 'verbose') else False
    print_verbose(verbose, f"unload_text_encoder() called")
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            print_verbose(verbose, f"Unloading {len(model.text_encoder)} text encoders from list")
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder"):
                print_verbose(verbose, f"Unloading text_encoder: moving from {pipe.text_encoder.device} to CPU")
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                pipe.text_encoder.to('cpu')
                print_verbose(verbose, f"text_encoder moved to CPU, replacing with FakeTextEncoder")
                pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                te_name = f"text_encoder_{i}"
                te_obj = getattr(pipe, te_name)
                print_verbose(verbose, f"Unloading {te_name}: moving from {te_obj.device} to CPU")
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                te_obj.to('cpu')
                print_verbose(verbose, f"{te_name} moved to CPU, replacing with FakeTextEncoder")
                setattr(pipe, te_name, te)
                i += 1
            model.text_encoder = text_encoder_list
            print_verbose(verbose, f"All text encoders unloaded, replaced {len(text_encoder_list)} encoders")
        else:
            # only has a single text encoder
            print_verbose(verbose, f"Unloading single text encoder from {model.text_encoder.device}")
            model.text_encoder = model.text_encoder.to('cpu')
            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            print_verbose(verbose, f"Text encoder moved to CPU and replaced with FakeTextEncoder")

    flush()
    print_verbose(verbose, f"Flushed GPU cache after unloading text encoders")
    print_verbose(verbose, f"unload_text_encoder() completed")
