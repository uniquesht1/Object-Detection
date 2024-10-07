# Selective Search and RCNN Implementation

This repository demonstrates object detection using **Selective Search** for region proposals and **RCNN (Region-based Convolutional Neural Networks)** for object classification and localization.

## 1. Selective Search

### Overview
Selective Search generates object-like region proposals, which are later passed to an RCNN model for object detection. It segments the image into candidate regions based on pixel similarity.

### Key Steps
- **Image Preprocessing**: Load and convert the image.
- **Selective Search**: Generate region proposals using multiple scales and size thresholds.
- **Region Extraction**: Filter and extract relevant bounding box regions.

### Code Snippets
- **Load Image**:
    ```python
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    ```
- **Run Selective Search**:
    ```python
    imgs, regions = selectivesearch.selective_search(img, scale=100, min_size=50)
    ```
- **Extract Candidates**:
    ```python
    def extract_candidates(img):
        _, regions = selectivesearch.selective_search(img, scale=100, min_size=50)
        candidates = [r['rect'] for r in regions if valid_region(r)]
        return candidates
    ```

## 2. RCNN (Region-based Convolutional Neural Networks)

### Overview
RCNN classifies object regions proposed by Selective Search and refines bounding boxes using regression. The backbone model (ResNet50) extracts features, while two heads handle classification and box regression.

### Key Steps
- **Data Loading**: Custom dataset loader for images and bounding boxes.
- **Model Architecture**: ResNet50 backbone, classification, and regression heads.
- **Training and Testing**: Loss computation for classification and bounding box regression, using IoU and Non-Maximum Suppression (NMS).

### Code Snippets
- **RCNN Model**:
    ```python
    class RCNN(nn.Module):
        def __init__(self, backbone, n_classes):
            super().__init__()
            self.backbone = backbone
            self.classification_head = nn.Linear(2048, n_classes)
            self.bbox_regression_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 4), nn.Tanh())
    ```
- **Training**:
    ```python
    for inputs, labels, deltas in train_dataloader:
        _labels, _deltas, loss, _, _ = train_batch(rcnn, optimizer, inputs, labels, deltas)
    ```
- **Prediction**:
    ```python
    def predict(inputs):
        with torch.no_grad():
            labels, deltas = rcnn(inputs)
            conf, clss = torch.softmax(labels, -1).max(-1)
        return conf, clss, deltas
    ```

## How to Run

### Prerequisites
- Python 3.x, OpenCV, PyTorch, Selective Search, TorchVision.

### Instructions
1. Clone the repo:
    ```bash
    git clone https://github.com/uniquesht1/Object-Detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Selective Search:

4. Train the RCNN:

5. Test predictions:


## Results
- Selective Search generates candidate regions, and the RCNN classifies objects and refines their bounding boxes using regression and NMS.

### Example Output
- Bounding box from Selective Search:
  ![image](https://github.com/user-attachments/assets/fbea7e3b-1c28-4289-9a41-c108a41f35d4)

- Original image and with bounding boxes from RCNN predictions:
  ![image](https://github.com/user-attachments/assets/c7d9b37f-4a03-40bc-b02a-60a56e2eb3fc)

- Image with bounding box from customly trained model.pth. It has low accuracy :)
  ![image](https://github.com/user-attachments/assets/aa947ff0-eebd-458c-9bba-98f22cefa9e2)

## License

This project is open-source and available under the MIT [LICENSE](LICENSE).
