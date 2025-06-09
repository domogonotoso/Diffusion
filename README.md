diffusion_project/
├── configs/
│   └── default.yaml            # Configuration file for training and sampling
├── data/
│   └── prepare_dataset.py      # Script to load and preprocess datasets
├── models/
│   └── unet.py                 # UNet model definition used in diffusion
├── diffusion/
│   ├── scheduler.py            # Beta scheduler for noise levels
│   └── diffusion.py            # Diffusion process (forward + reverse)
├── train.py                    # Main training script
├── sample.py                   # Image sampling script using trained model
├── utils/
│   ├── logger.py               # Logging utilities (e.g., wandb, stdout)
│   └── visualizer.py           # Utility to save or display generated images
├── samples/                    # Directory to store generated images
├── checkpoints/                # Directory to save model checkpoints
├── requirements.txt            # Python package dependencies
└── README.md                   # Project overview and instructions




Concept
schedular process, noise and denoise process with formular


AI image generator is popular like Stable Diffusion even to people who doesn't get used to AI tech.
But this time I just imply simple diffusion model that U-net without text prompt. 


Later, next project(stable diffusion) classifier guidance,    Classifier-Free Guidance(CFG)!!!!!