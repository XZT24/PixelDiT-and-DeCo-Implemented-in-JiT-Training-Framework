
<br>
<p align="center">
  <h1 align="center"><strong>PixelDiT and DeCo Implemented in JiT Training Framework</strong></h1>
  </p>
</p>

## ⭐ Overview
We implement two advanced pixel-space diffusion models, [PixelDiT](https://github.com/NVlabs/PixelDiT) and [DeCo](https://github.com/Zehong-Ma/DeCo), within the [JiT](https://github.com/LTH14/JiT) training framework. 
Compared with the official implementations, this code features a cleaner and more streamlined structure, making it easier to further modify and optimize.

The network architectures and model configurations remain consistent with their official open-source implementations, while the training settings have been carefully tuned to ensure stable training. 
Compared with the official implementations, we replace [REPA](https://github.com/sihyun-yu/REPA) with an improved version of [iREPA](https://github.com/end2end-diffusion/irepa) and adopt the same \(t\)-sampling strategy as JiT to facilitate fair comparisons across different models.
Check the codes for more details.
   
---

## 🛠️ Environment Setup
Follow the exact same environment setup procedure as [JiT](https://github.com/LTH14/JiT).
```bash
conda env create -f environment.yaml
conda activate jit
```

---

## 📚 Getting Started
**Recommended Settings:** Training PixelDiT-XL/16 requires approximately 45 GB of GPU memory. We recommend using GPUs with at least 40 GB of memory, such as the A100. 
If memory consumption is too high, you can reduce the batch size accordingly. The learning rate will be automatically scaled to maintain comparable training performance.

Training: 
```bash
sh train.sh
```
Evaluation: 
```bash
sh eval.sh
```
---

## 📬 Contact
If you have questions, feel free to open an issue.

---

## 🔗 Citation
We thank the authors of [DeCo](https://github.com/Zehong-Ma/DeCo), [PixelDiT](https://github.com/NVlabs/PixelDiT), and [JiT](https://github.com/LTH14/JiT) for making their code publicly available. The corresponding BibTeX citations are provided below.

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}

@article{ma2025decofrequencydecoupledpixeldiffusion,
      title={DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation}, 
      author={Zehong Ma and Longhui Wei and Shuai Wang and Shiliang Zhang and Qi Tian},
      year={2025},
      eprint={2511.19365},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19365}, 
}

@inproceedings{yu2026pixeldit,
      title={PixelDiT: Pixel Diffusion Transformers for Image Generation},
      author={Yongsheng Yu and Wei Xiong and Weili Nie and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2026},
}
