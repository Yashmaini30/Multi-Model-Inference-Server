# Multi-Model Inference Server

## Overview

The Multi-Model Inference Server is a Flask application designed for performing image classification using multiple pre-trained models. It supports the following models:

- **DenseNet121**
- **ResNet50**
- **AlexNet**

The server provides endpoints to make predictions with each model and includes detailed logging for operational insights.

## Features

- **Multi-Model Support**: Infer using DenseNet121, ResNet50, and AlexNet.
- **Logging**: Detailed logging for debugging and operational monitoring.
- **Image Preprocessing**: Includes resizing and normalization of images.

## Getting Started

### Prerequisites

- **Docker**: For containerizing the application.
- **Python 3.11**: For local development and running the Flask application.
- **Model Files**: Ensure that the model files (`.pth`) are available in the `models/` directory.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Yashmaini30/multi-model-inference-server.git
   cd multi-model-inference-server
   ```

2. **Build the Docker Image**

   ```bash
   docker build -t ghcr.io/yashmaini30/multi-model-inference-server:latest .
   ```

3. **Run the Docker Container**

   ```bash
   docker run -p 5000:5000 ghcr.io/yashmaini30/multi-model-inference-server:latest
   ```

4. **Access the Server**

   Open your browser or use tools like `curl` or Postman to interact with the API at `http://localhost:5000`.

## Usage

### Predict with DenseNet121

**Endpoint**: `/predict/densenet`  
**Method**: POST  
**Content-Type**: `multipart/form-data`  
**Parameters**:
- `file`: The image file to be classified.

**Example Request**:

```bash
curl -X POST -F "file=@path_to_your_image.jpg" http://localhost:5000/predict/densenet
```

**Response**:

```json
{
  "message": "Prediction using DenseNet121",
  "prediction": [[0.1, 0.7, 0.2]]  // Example prediction probabilities
}
```

### Predict with ResNet

**Endpoint**: `/predict/resnet`  
**Method**: POST  
**Content-Type**: `multipart/form-data`  
**Parameters**:
- `file`: The image file to be classified.

**Example Request**:

```bash
curl -X POST -F "file=@path_to_your_image.jpg" http://localhost:5000/predict/resnet
```

**Response**:

```json
{
  "message": "Prediction using ResNet",
  "prediction": [[0.2, 0.6, 0.2]]  // Example prediction probabilities
}
```

### Predict with AlexNet

**Endpoint**: `/predict/alexnet`  
**Method**: POST  
**Content-Type**: `multipart/form-data`  
**Parameters**:
- `file`: The image file to be classified.

**Example Request**:

```bash
curl -X POST -F "file=@path_to_your_image.jpg" http://localhost:5000/predict/alexnet
```

**Response**:

```json
{
  "message": "Prediction using AlexNet",
  "prediction": [[0.3, 0.4, 0.3]]  // Example prediction probabilities
}
```

### Error Handling

Errors are logged and returned in the HTTP response. Common errors include:

- **Model Loading Errors**: Issues when loading the model weights.
- **Image Processing Errors**: Issues with image preprocessing.
- **Prediction Errors**: Issues during inference.

### Logging

The application uses Python's built-in logging module to record:

- Model loading errors
- Image processing errors
- Prediction errors

Logs are output to the console by default, but can be configured as needed.

## Docker Integration

### Build Docker Image

To build the Docker image, use:

```bash
docker build -t ghcr.io/yashmaini30/multi-model-inference-server:latest .
```

### Push Docker Image

To push the Docker image to GHCR:

```bash
docker push ghcr.io/yashmaini30/multi-model-inference-server:latest
```

### Pull Docker Image

To pull the Docker image from GHCR:

```bash
docker pull ghcr.io/yashmaini30/multi-model-inference-server:latest
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

