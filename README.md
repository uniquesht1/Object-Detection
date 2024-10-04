# RCNN Object Detection Model

This repository implements a **Region-based Convolutional Neural Network (RCNN)** for object detection using **PyTorch** and **selective search**. The model is trained to classify objects and localize bounding boxes in images.

## Features

- **Selective Search**: To propose candidate regions from the input images.
- **ResNet50 Backbone**: Used for feature extraction.
- **Bounding Box Regression**: Localizes objects by predicting bounding box coordinates.
- **Classification Head**: Classifies the objects detected in the bounding boxes.
- **Model Training**: Custom training loop with support for batch processing and model evaluation.
- **Inference Pipeline**: Includes preprocessing, forward pass, post-processing, and visualization.

## Dependencies

- Python 3.x
- PyTorch
- OpenCV
- pandas, numpy
- tqdm
- matplotlib
- selectivesearch

Install dependencies using:

```bash
pip install torch torchvision opencv-python pandas numpy tqdm matplotlib selectivesearch
```

## How to Use

1. **Prepare Dataset**:
   * Update the `image_paths` and `csv_path` to point to your dataset of images and annotations (CSV file).
2. **Train the Model**:
   * Run the notebook to load and preprocess the dataset, initialize the model, and start training.
3. **Save the Model**:
   * After training, save the model or its `state_dict` for future inference.
4. **Run Inference**:
   * Use the `InferenceRCNN` class to perform object detection on new images using the trained model.

Example:

```python
inference = InferenceRCNN(test, device, background_class, preprocess, nms, extract_candidates)
inference('path_to_image.jpg', display=True)
```

## Example Usage

To train the model:

```python
rcnn = RCNN(backbone, n_classes=len(unique_labels)).to(device)
optimizer = torch.optim.SGD(rcnn.parameters(), lr=learning_rate)
train_batch(rcnn, optimizer, inputs, labels, deltas)
```

To run inference after loading the model:

```python
test = torch.load('model.pth', map_location=device)
test.eval()
inference = InferenceRCNN(test, device, background_class, preprocess, nms, extract_candidates)
inference('test_image.jpg', display=True)
```

## Model Architecture

* **Backbone**: ResNet50 (pretrained on ImageNet).
* **Classification Head**: Linear layer for class prediction.
* **Localization Head**: Regression head to predict bounding box coordinates.

## Project Structure

* **data/**: Folder containing images and CSV files for training.
* **notebook.ipynb**: Main notebook with the full pipeline (data loading, model definition, training, and inference).
* **model.pth**: Trained model saved using `torch.save()` (not included by default).

## License

This project is open-source and available under the MIT License.
