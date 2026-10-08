# Image Classifier
Train a CNN to classify images stored in class-specific folders. `main.py` trains and evaluates the model; `image.py` runs a sample prediction.
## Dataset
Place images in subfolders under `dataset/`:
```text
dataset/
├── Cat/
└── Dog/
```
Folder names define the classes. Training supports two or more classes, resizes images to 160 × 160, and uses a 70/30 training/validation split. Training images are augmented.
## Install and run
From the repository root, install the required packages:
```bash
python -m pip install tensorflow opencv-python pillow matplotlib
```
Train the model:
```bash
python main.py
```
The trained model is saved as `image_classifier.keras`. To run the prediction example:
```bash
python image.py
```
## Notes
- `image.py` requires `dataset/Cat/` and `dataset/Dog/`, files named from `0.jpg` to `12499.jpg`, and a trained model. It selects an image randomly; it does not accept an image path as an argument.
- Class names are read in a different way during prediction than during training, so the predicted label may not match the model's class order.
- Evaluation uses the validation set; there is no separate test set. The repository does not pin dependency versions.
