# Citation & use

Birdbrain is intended for **non-commercial research and educational purposes only**. It is not licensed for commercial deployment or redistribution of the underlying datasets. If you use this project or its training data in published work, please cite the sources below and review each dataset's terms on the official sites.

## CUB-200-2011 dataset

Production models and the current web classifier are trained on the [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset.

> **The Caltech-UCSD Birds-200-2011 Dataset**  
> Wah, C.; Branson, S.; Welinder, P.; Perona, P.; Belongie, S.  
> California Institute of Technology, 2011.  
> Technical Report CNS-TR-2011-001

**License:** CUB images are restricted to non-commercial research and educational use. See the [official dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/) for terms.

### BibTeX (CUB)

```bibtex
@techreport{WelinderEtal2010,
  Author = {C. Wah and S. Branson and P. Welinder and P. Perona and S. Belongie},
  Title = {Caltech-UCSD Birds-200-2011},
  Institution = {California Institute of Technology},
  Year = {2011},
  Number = {CNS-TR-2011-001}
}
```

## iNaturalist 2021 dataset

iNat training experiments use the [iNat Challenge 2021](https://kaggle.com/competitions/inaturalist-2021) species classification data (FGVC8). Birdbrain's `INat2021Dataset` filters to **Aves** (birds) only.

> Grant Van Horn and macaodha. **iNat Challenge 2021 - FGVC8**.  
> https://kaggle.com/competitions/inaturalist-2021, 2021. Kaggle.

Additional reference: [Visipedia iNat competition repository](https://github.com/visipedia/inat_comp).

**License / use:** Follow the Kaggle competition and iNaturalist dataset terms. This project uses the data for research and educational fine-tuning only—not for commercial products or redistribution of the raw images.

### BibTeX (iNat 2021)

```bibtex
@misc{inaturalist2021,
  author = {Grant Van Horn and macaodha},
  title = {iNat Challenge 2021 - {FGVC8}},
  howpublished = {\url{https://kaggle.com/competitions/inaturalist-2021}},
  year = {2021},
  publisher = {Kaggle}
}
```

## Model weights

Checkpoints use ImageNet-pretrained backbones from [torchvision](https://pytorch.org/vision/stable/models.html) (EfficientNet-B0, ResNet50). Cite torchvision and the original backbone papers if you publish results that depend on those weights.

## This project

If you reference the Birdbrain training pipeline or web demo itself, link to this repository and note which dataset(s) and checkpoint stage(s) you used (CUB-only, iNat birds-only, or both).
