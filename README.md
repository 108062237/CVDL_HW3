# HW3: Medical Cell Instance Segmentation

This project uses the Mask R-CNN model to perform instance segmentation on four types of cells in medical microscopy images. The codebase supports training, inference, submission generation, and visualization. It is modular and easy to reproduce experiments.

## Instructions

### 1. Environment Setup

Go to the `src/` directory and run:

Using Conda:
```
conda env create -f environment.yml
conda activate hw3-env
```

### 2. Data Preparation

Place your dataset under the `src/data/` folder with the following structure:

```
data/
├── train/
│   └── [image_id]/
│       ├── image.tif
│       ├── class1.tif
│       ├── class2.tif
│       ├── class3.tif
│       └── class4.tif
├── test/
│   └── [image_id].tif
└── test_image_name_to_ids.json
```

Each `classX.tif` contains pixel-wise instance IDs for that class (0 = background, >0 = instance N).

### 3. Train the Model

Run the following command to start training:

```
python train.py --config config/config.yaml 
```

The trained model will be saved to `checkpoints/best.pth`.

### 4. Generate Submission

After training is complete, run:

```
python submission.py --config config/config.yaml
```

This script loads the trained model, performs inference on the test set, and outputs a `test-results.json` file in COCO format.  
You can upload this file to the CodaLab or CodaBench competition platform.

## Code Overview

- `dataloader.py`: Defines the CellDataset and DataLoader logic.  
- `model.py`: Builds the Mask R-CNN model with a configurable backbone.  
- `train.py`: Training loop with support for various optimizers and schedulers.  
- `submission.py`: Inference script that produces the final JSON output.  
- `config/config.yaml`: Model configuration and training hyperparameters.  
- `environment.yml`: Conda environment specification.  
