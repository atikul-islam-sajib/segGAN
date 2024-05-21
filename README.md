# seGAN(Segmentation GAN - Image to Image translation) Project

<img src="https://github.com/atikul-islam-sajib/Research-Assistant-Work-/blob/main/download%20(1).png">

This project provides a complete framework for training and testing a Segmentation Generative Adversarial Network (SeGAN). It includes functionality for data preparation, model training, testing, and inference to enhance low-resolution images to image translation(seGAN)

<img src="https://www.researchgate.net/publication/317378234/figure/fig4/AS:962701794762773@1606537402648/The-architecture-of-the-proposed-SegAN-with-segmentor-and-critic-networks-In-the.gif" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized seGAN model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of seGAN functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/segGAN.git** |
| 2    | Navigate into the project directory.         | **cd segGAN**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the srgan model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the seGAN model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for srgan

The dataset is organized into three categories for seGAN. Each category directly contains paired images and their corresponding lower resolution images and higher resolution, stored together to simplify the association between lower resolution and higher resolution images .

## Directory Structure:

```
datasets/
├── images/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── masks/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

For detailed documentation on the dataset visit the [Dataset - Kaggle](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery).



### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--smooth`  | Smooth of Dice loss                       | float  | 0.001   |
| `--lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--is_weight_init`| Apply weight initialization                  | bool   | False   |
| `--is_display`    | Display detailed loss information            | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--beta1`         | Beta1 parameter for Adam optimizer           | float  | 0.5     |
| `--beta2`         | Beta1 parameter for Adam optimizer           | float  | 0.999     |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--test`          | Flag to initiate testing mode                | action | N/A     |

### CLI Command Examples

To train or test the model using a specified configuration file, use the following commands:

| Command                                 | Description                                |
|-----------------------------------------|--------------------------------------------|
| `python -m src.cli --config ./config.yml --train` | Train the model using the specified configuration file. |
| `python -m src.cli --config ./config.yml --test`  | Test the model using the specified configuration file.  |

## Examples

### Training
```sh
python -m src.cli --config ./config.yml --train
```

### Testing
```sh
python -m src.cli --config ./config.yml --test
```

#### Manual Configuration

For more advanced usage, you can manually configure the training and testing commands. Here are examples for different setups:

| Task                  | Command                                                                                                                                                             |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**  | `python -m src.cli --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --l1loss_value 0.01 --adam True --device "cuda"`   |
| **Testing a Model**   | `python -m src.cli --test --device "cuda"`                                                                                                                           |

> **Note:** If you are using a device other than CUDA (e.g., MPS or CPU), simply replace `--device "cuda"` with `--device "mps"` or `--device "cpu"` in the command.

These commands allow for precise control over various parameters to optimize training and testing performance according to your specific requirements.

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **MPS Command**: For Apple Silicon (M1, M2 chips), using the `mps` device can provide optimized performance.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.


#### Initializing Data Loader - Custom Modules
```python
from src.dataloader import Loader

loader = Loader(image_path="path/to/dataset", batch_size=32, image_size=128)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
print(loader.details_dataset())   # It will give a CSV file about dataset
loader.plot_images()              # It will display the images from dataset
```

#### Training the Model
```python
from src.trainer import Trainer

trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer
    l1loss_value 0.01,         # Weight for content loss in the loss calculation
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    beta1=0.5,                 # Beta1 parameter for Adam optimizer
    lr_scheduler=False,        # Enable a learning rate scheduler
    is_weight_init=False,      # Enable custom weight initialization for the models
    is_display=True            # Display training progress and statistics
                               # Explore other parameters in the Trainer class documentation ....
)

# Start training
trainer.train()
```

##### Training Performances
```python
print(trainer.plot_history())    # It will plot the netD and netG losses for each epochs
```

#### Testing the Model
```python
from src.test import TestModel

test = TestModel(device="cuda")  # use "mps", "cpu"
test.plot()
```


## Contributing
Contributions to improve this implementation of seGAN are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).
