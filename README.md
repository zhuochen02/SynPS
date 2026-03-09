# [CVPR 2026] The Devil is in Attention Sharing: Improving Complex Non-rigid Image Editing Faithfulness via Attention Synergy

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-2512.14423-b31b1b.svg)](https://arxiv.org/abs/2512.14423)
[![Project Page](https://img.shields.io/badge/Project_Page-synps26.github.io-green)](https://synps26.github.io/)

**SynPS** (**Syn**ergistically leverages **P**ositional embeddings and **S**emantic information) is a training-free method for complex non-rigid image editing. By synergistically integrating positional embeddings and semantic information, it improves the faithfulness of edits with large diffusion models (e.g., FLUX). Published at **CVPR 2026**.

📄 **Paper** [arXiv:2512.14423](https://arxiv.org/abs/2512.14423) | 🌐 **Project Page** [synps26.github.io](https://synps26.github.io/)

**Authors**: Zhuo Chen, Fanyue Wei, Runze Xu, Jingjing Li, Lixin Duan, Angela Yao, Wen Li

## Method Overview

Existing attention sharing mechanisms suffer from **attention collapse**: either positional embeddings or semantic features dominate visual content retrieval, leading to over-editing or under-editing. SynPS addresses this by:

1. **Editing Measurement**: Quantifying the required editing magnitude at each denoising step;
2. **Attention Synergy Pipeline**: Dynamically modulating the influence of positional embeddings based on this measurement to balance semantic modifications and fidelity preservation;
3. **Adaptive Integration**: Scheduling `pe_weight` across timesteps so SynPS adaptively switches between positional and semantic cues, effectively avoiding both over- and under-editing.


## Setup

### Dependencies

Create environment with Python 3.10, then install:

```bash
pip install -r requirements.txt
```

### Model Download

This project uses the **FLUX.1-dev** model. As it is a gated repository, you need to accept the license and log in to Hugging Face:

1. Visit [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Log in to your Hugging Face account and accept the license terms
3. Authenticate via one of:
   ```bash
   huggingface-cli login
   ```
   Or set the environment variable:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

Models will be downloaded to `./checkpoint` by default. You can override this with the `CHECKPOINTS_DIR` environment variable.

## Quick Start

### Jupyter Notebook

We recommend using `demo.ipynb` for a quick demo:

```bash
jupyter notebook demo.ipynb
```

### Configuration

You can adjust the following parameters in `demo.ipynb`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | `flux-dev` | Model name |
| `guidance` | `3.5` | Classifier-free guidance scale |
| `num_steps` | `50` | Number of denoising steps |
| `pe_threshold_max` | `1.0` | Upper threshold for PE weight scheduling |
| `pe_threshold_min` | `0.9` | Lower threshold for PE weight scheduling |
| `output_dir` | `./results` | Output directory |
| `offload` | `False` | Offload models to CPU to save VRAM |


## Project Structure

```
SynPS/
├── README.md
├── demo.ipynb          # Demo notebook
├── flux/
│   ├── __init__.py
│   ├── model.py        # Flux model with pe_cross and pe_weight
│   ├── math.py         # Attention and RoPE; cross-batch KV replacement logic
│   ├── sampling.py     # Denoising loop and pe_weight dynamic scheduling
│   ├── util.py         # Model loading, checkpoint management, etc.
│   └── modules/
│       ├── layers.py   # DoubleStreamBlock, SingleStreamBlock; attn_similarity computation
│       ├── autoencoder.py
│       ├── conditioner.py
│       └── ...
└── ...
```

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{chen2025synps,
  title={The Devil is in Attention Sharing: Improving Complex Non-rigid Image Editing Faithfulness via Attention Synergy},
  author={Chen, Zhuo and Wei, Fanyue and Xu, Runze and Li, Jingjing and Duan, Lixin and Yao, Angela and Li, Wen},
  journal={arXiv preprint arXiv:2512.14423},
  year={2025}
}
```

arXiv: [2512.14423](https://arxiv.org/abs/2512.14423) | DOI: [10.48550/arXiv.2512.14423](https://doi.org/10.48550/arXiv.2512.14423)

## Acknowledgments

- This method is built on [FLUX](https://github.com/black-forest-labs/flux) and [FreeFlux](https://github.com/wtybest/FreeFlux)
- Model weights are from [Black Forest Labs](https://blackforestlabs.ai/)

## License

Please adhere to the license terms of the FLUX model and Black Forest Labs when using this repository and pretrained weights. For commercial licensing, see [BFL Licensing](https://blackforestlabs.ai/).
