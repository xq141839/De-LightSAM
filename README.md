# TCSVT 2025 | De-LightSAM: Modality-Decoupled Lightweight SAM for Generalizable Medical Segmentation

:pushpin: This is an official PyTorch implementation of **De-LightSAM: Modality-Decoupled Lightweight SAM for Generalizable Medical Segmentation**

[[`arXiv`](https://arxiv.org/abs/2407.14153)] [[`BibTeX`](https://scholar.googleusercontent.com/scholar.bib?q=info:m1SgUw239yoJ:scholar.google.com/&output=citation&scisdr=ClFw2KpjEKieyAzytL0:AFWwaeYAAAAAZrn0rL2ppFHGbu1Xxj0lav_6C68&scisig=AFWwaeYAAAAAZrn0rNOsqsq5rY17-SQIAK7bscM&scisf=4&ct=citation&cd=-1&hl=en)]


<div align="center">
    <img width="100%" alt="De-LightSAM overview" src="fig_method.png"/>
</div>

<div align="center">
    <img width="100%" alt="De-LightSAM overview" src="fig_method_kd.png"/>
</div>

## 📰News

**[2025.10.11]** The article has been accepted by: IEEE Transactions on Circuits and Systems for Video Technology.

**[2024.08.08]** The pre-print paper has been uploaded!

**[2024.08.07]** Paper will be updated soon!

**[2024.08.07]** Code and model checkpoints are released!

## 🛠Setup

```bash
git clone https://github.com/xq141839/De-LightSAM.git
cd De-LightSAM
conda create -n ESP python=3.10
conda activate ESP
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install albumentations==0.5.2
pip install pytorch_lightning==1.1.1
pip install monai
```

**Note**: Please refer to requirements.txt


## 📚Data Preparation

The structure is as follows.
```
De-LightSAM
├── datasets
│   ├── image_1024
│     ├── ISIC_0000000.png
|     ├── ...
|   ├── mask_1024
│     ├── ISIC_0000000.png
|     ├── ...
```

## 🎪Segmentation Model Zoo
We provide all pre-trained models here.
| MA-Backbone | MC | Checkpoints |
|-----|------|-----|
|TinyViT| Dermoscopy | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|
|TinyViT| X-ray | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|
|TinyViT| Fundus | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|
|TinyViT| Colonoscopy | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|
|TinyViT| Ultrasound | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|
|TinyViT| Microscopy | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|

## 📜Citation
If you find this work helpful for your project, please consider citing the following paper:
```
@article{xu2024esp,
  title={ESP-MedSAM: Efficient Self-Prompting SAM for Universal Domain-Generalized Medical Image Segmentation},
  author={Xu, Qing and Li, Jiaxuan and He, Xiangjian and Liu, Ziyu and Chen, Zhen and Duan, Wenting and Li, Chenxin and He, Maggie M and Tesema, Fiseha B and Cheah, Wooi P and others},
  journal={arXiv preprint arXiv:2407.14153},
  year={2024}
}
```
