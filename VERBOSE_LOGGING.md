# Verbose Logging for LTX-2 Models

This documentation describes the verbose logging feature added to track LTX-2 model operations during training and inference.

## Overview

Verbose logging provides detailed output about:
- Model loading and device movements
- VAE and audio encoding operations
- Text encoder operations
- Transformer forward passes
- Training loop iterations
- Optimizer steps
- Memory management operations
- File I/O operations

## Enabling Verbose Logging

To enable verbose logging, add the `verbose: true` flag to your model configuration:

```yaml
model:
  name_or_path: "path/to/ltx2/model"
  arch: "ltx2"
  verbose: true  # Enable verbose logging
  # ... other model config options
```

## Log Output Format

All verbose logs are prefixed with `[VERBOSE]` to make them easily identifiable in the output:

```
[VERBOSE] load_model() called with dtype=torch.bfloat16
[VERBOSE] Model paths: model_path=/path/to/model, base_model_path=/path/to/base
[VERBOSE] Loading transformer from pretrained: path=/path/to/transformer, subfolder=transformer
[VERBOSE] Transformer loaded from pretrained
```

## What Gets Logged

### Model Loading (`load_model`)
- Model paths and configuration
- Component loading (transformer, text encoder, VAE, etc.)
- Device movements
- Quantization operations
- Layer offloading setup
- Memory manager attachment

### Image Encoding (`encode_images`)
- Number of images being encoded
- Device and dtype information
- VAE device movements
- Image shape transformations
- Latent normalization

### Audio Encoding (`encode_audio`)
- Number of audio items
- Audio VAE device movements
- Waveform shape transformations
- Mel spectrogram conversion
- Latent packing operations

### Noise Prediction (`get_noise_prediction`)
- Latent dimensions and timesteps
- Image-to-video conditioning
- Audio processing
- Connector operations
- Transformer forward pass
- Output unpacking

### Prompt Encoding (`get_prompt_embeds`)
- Batch size and prompts
- Text encoder device movements
- Tokenization details
- Hidden states processing
- Embedding packing

### Generation (`generate_single_image`)
- Generation parameters (frames, dimensions)
- Pipeline switching (text-to-video vs image-to-video)
- Control image processing
- VAE tiling (low VRAM mode)
- Inference steps and guidance scale
- Output processing

### Training Loop (`hook_train_loop`)
- Step number
- Number of batches
- Gradient accumulation
- Loss per batch
- Optimizer steps
- Learning rate scheduler
- EMA updates
- Average loss

### Model Prediction (`predict_noise`)
- Latent shapes and timesteps
- Primary vs auxiliary predictions
- Result shapes

### Text Encoder Unloading (`unload_text_encoder`)
- Number of encoders being unloaded
- Device movements to CPU
- FakeTextEncoder replacements

### Memory Management (`MemoryManager.attach`)
- Module names and devices
- Offload percentages
- Layer counts (linear, conv, unmanaged)

### Model Saving (`save_model`)
- Output paths
- File operations

## Performance Considerations

Verbose logging adds minimal overhead to training and inference. The logs are:
- Only printed on the main process (multi-GPU aware)
- Disabled by default (opt-in)
- Non-blocking I/O operations

However, for production training runs, you may want to disable verbose logging after initial debugging.

## Examples

### Training Configuration with Verbose Logging

```yaml
job: extension
config:
  name: ltx2_training_verbose
  process:
    - type: sd_trainer
      training_folder: "output/ltx2_training"
      
      model:
        name_or_path: "Lightricks/LTX-2"
        arch: "ltx2"
        verbose: true  # Enable verbose logging
        
      train:
        steps: 1000
        gradient_accumulation: 4
        # ... other training config
        
      sample:
        sample_every: 100
        # ... sampling config
```

### Output Example

```
Loading LTX2 model
[VERBOSE] load_model() called with dtype=torch.bfloat16
[VERBOSE] Model paths: model_path=Lightricks/LTX-2, base_model_path=Lightricks/LTX-2
Loading transformer
[VERBOSE] Starting text encoder loading
[VERBOSE] Loading transformer from pretrained: path=Lightricks/LTX-2, subfolder=transformer
[VERBOSE] Transformer loaded from pretrained
...
[VERBOSE] hook_train_loop() called at step 0
[VERBOSE] Processing 4 batches for gradient accumulation
[VERBOSE] Training batch 1/4
[VERBOSE] predict_noise() called (primary prediction): latents shape=torch.Size([1, 32, 9, 64, 64]), timesteps=tensor([...])
[VERBOSE] get_noise_prediction() called: latent_input shape=torch.Size([1, 32, 9, 64, 64]), timesteps=tensor([...])
...
[VERBOSE] Batch 1 loss: 0.0234
[VERBOSE] hook_train_loop() completed with average loss: 0.0245
```

## Troubleshooting

If verbose logging is not appearing:

1. **Check that `verbose: true` is set in the model config** (not the logging config)
2. **Verify you're on the main process** - verbose logs only show on the main process in distributed training
3. **Check your output redirection** - ensure stdout is not being suppressed

## Related Configuration

The `verbose` flag in `LoggingConfig` is separate and controls different aspects:
- `model.verbose` - Controls model-level verbose logging (what this document describes)
- `logging.verbose` - Controls general logging framework verbosity

For LTX-2 model debugging, you want `model.verbose: true`.
