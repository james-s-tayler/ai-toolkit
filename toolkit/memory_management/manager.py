import torch
from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager
import random
from toolkit.print import print_verbose

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []

    def memory_managed_to(self, *args, **kwargs):
        # first move all the unmanaged modules
        for module in self.unmanaged_modules:
            if isinstance(module, torch.nn.Parameter):
                # Parameter cannot move this way
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls, 
        module: torch.nn.Module, 
        device: torch.device, 
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = [],
        verbose: bool = False
    ):
        print_verbose(verbose, f"MemoryManager.attach() called for module {module.__class__.__name__}, device={device}, offload_percent={offload_percent}")
        if hasattr(module, "_memory_manager"):
            # already attached
            print_verbose(verbose, f"MemoryManager already attached to {module.__class__.__name__}, skipping")
            return

        module._memory_manager = cls(module, device)
        print_verbose(verbose, f"MemoryManager instance created for {module.__class__.__name__}")

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to
        print_verbose(verbose, f"Overridden .to() method for {module.__class__.__name__}")

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)
        print_verbose(verbose, f"Added {len(ignore_modules)} modules to ignore list")
            
        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]
        linear_count = 0
        conv_count = 0
        unmanaged_count = 0
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                        unmanaged_count += 1
                    else:
                        # linear
                        LinearLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        linear_count += 1
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara, 
                                    device,
                                    verbose=verbose
                                )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                        unmanaged_count += 1
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        conv_count += 1
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara, 
                                    device,
                                    verbose=verbose
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                    unmanaged_count += 1
                else:
                    continue
        print_verbose(verbose, f"MemoryManager.attach() completed for {module.__class__.__name__}: {linear_count} linear layers, {conv_count} conv layers, {unmanaged_count} unmanaged modules")
