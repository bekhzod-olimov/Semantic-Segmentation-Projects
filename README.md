# Semantic-Segmentation-Projects

This repository contains sample semantic segmentation projects using open source datasets and [PyTorch Lightning library](https://www.pytorchlightning.ai/index.html).

## Dependencies

* Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
* Activate the environment using the following command:
```python
conda activate speed
```

## Train
* Run train script
```python
python main.py --dataset_name drone --model_name unet --devices 4 --epochs 50
```

## Inference
* Run inference script
```python
python inference.py --dataset_name drone --model_name unet
```

## Inference Results
* [Flood Area Segmentation Dataset](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)

**[UNet](https://arxiv.org/abs/1505.04597)**

![flood_unet_preds](https://github.com/bekhzod-olimov/Semantic-Segmentation-Projects/assets/50166164/34557831-6498-41fa-84e7-7fa939b80cfc)

**[SegFormer](https://arxiv.org/abs/2105.15203)**

![flood_segformer_preds](https://github.com/bekhzod-olimov/Semantic-Segmentation-Projects/assets/50166164/007053a7-2e69-4f6d-bacc-52ee362475c3)

* [Cells Segmentation Dataset](https://drive.google.com/file/d/1c4oON03uBSxcGlluBFHTtkhFibUPSWs7/view)

**[UNet](https://arxiv.org/abs/1505.04597)**

![cells_unet_preds](https://github.com/bekhzod-olimov/Semantic-Segmentation-Projects/assets/50166164/a98371c9-c023-4590-9873-aece8ca233b5)

**[SegFormer](https://arxiv.org/abs/2105.15203)**

![cells_segformer_preds](https://github.com/bekhzod-olimov/Semantic-Segmentation-Projects/assets/50166164/8bcc583d-22cf-4733-bb0b-3a3080e1ff55)

* [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)

**[UNet](https://arxiv.org/abs/1505.04597)**

![drone_unet_preds](https://github.com/bekhzod-olimov/Semantic-Segmentation-Projects/assets/50166164/c3f13d69-ecc7-409d-b828-acec764e169a)

**[SegFormer](https://arxiv.org/abs/2105.15203)**

![drone_segformer_preds](https://github.com/bekhzod-olimov/Semantic-Segmentation-Projects/assets/50166164/99375aa1-34aa-4bad-b37e-ce6bedcf2218)


