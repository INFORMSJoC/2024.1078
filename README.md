[![INFORMS Journal on Computing](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# RLGD

Standalone code for *Reinforcement Learning-based Graph Downsampling for Social Recommendation*, by Dongyang Li, Jianshan Sun, Xin Li, Chongming Gao, Fuli Feng, and Kun Yuan. This package contains the **Douban-Book** experiment and is prepared for the IJOC archive. The software is distributed under the [MIT License](LICENSE).

## Cite

Please cite both the paper and this repository:

- Paper: https://doi.org/10.1287/ijoc.2024.1078
- Code: https://doi.org/10.1287/ijoc.2024.1078.cd

```bibtex
@misc{Li2026RLGD,
  author = {Li, Dongyang and Sun, Jianshan and Li, Xin and Gao, Chongming and Feng, Fuli and Yuan, Kun},
  publisher = {INFORMS Journal on Computing},
  title = {Reinforcement Learning-based Graph Downsampling for Social Recommendation},
  year = {2026},
  doi = {10.1287/ijoc.2024.1078.cd},
  url = {https://github.com/INFORMSJoC/2024.1078},
  note = {Available for download at https://github.com/INFORMSJoC/2024.1078}
}
```

## Contents

- `src/`: training entry point and the standalone `cgsr` implementation.
- `data/`: processed Douban-Book interactions and social links.
- `results/`: Douban-Book experimental results.

## Run

Use Python 3.8. Install PyTorch 1.11.0 and the matching PyG extension wheels for your CPU/CUDA build using the [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/2.1.0/notes/installation.html), then install the remaining pinned dependencies:

```bash
python src/main.py --gpu_id=0
```

Change `--gpu_id` to select a physical GPU; `--gpu_id=""` uses the CPU.

Training uses a per-user 7:1:2 split, up to 200 epochs, and validation Recall@10 for model selection. New results are written to `results/Douban-Book/`.

