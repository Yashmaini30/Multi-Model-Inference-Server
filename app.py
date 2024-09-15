from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model architectures and paths
MODEL_PATHS = {
    'densenet121': 'models/densenet121.pth',
    'resnet': 'models/resnet50.pth',
    'alexnet': 'models/alexnet.pth'
}

def load_model(model_name):
    try:
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=False, num_classes=3)
        elif model_name == 'resnet':
            model = models.resnet50(pretrained=False, num_classes=3)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=False, num_classes=3)
        else:
            raise ValueError("Invalid model name")

        model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

# Load models
models_dict = {}
try:
    for model_name in MODEL_PATHS.keys():
        models_dict[model_name] = load_model(model_name)
except RuntimeError as e:
    logger.error(f"Model loading error: {e}")
    exit(1)

# Define image preprocessing function
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise RuntimeError(f"Error processing image: {str(e)}")

# Define prediction function
def predict(model, image_bytes):
    try:
        input_data = preprocess_image(image_bytes)
        with torch.no_grad():
            prediction = model(input_data)
        prediction = torch.softmax(prediction, dim=1).tolist()
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Error during prediction: {str(e)}")

# Route for the root URL
@app.route('/', methods=['GET'])
def index():
    return "Multi Model Inference Server"

# Route for inference using DenseNet121
@app.route('/predict/densenet', methods=['POST'])
def predict_densenet():
    try:
        image = request.files['file'].read()
        prediction = predict(models_dict['densenet121'], image)
        return jsonify({'message': 'Prediction using DenseNet121', 'prediction': prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route for inference using ResNet
@app.route('/predict/resnet', methods=['POST'])
def predict_resnet():
    try:
        image = request.files['file'].read()
        prediction = predict(models_dict['resnet'], image)
        return jsonify({'message': 'Prediction using ResNet', 'prediction': prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route for inference using AlexNet
@app.route('/predict/alexnet', methods=['POST'])
def predict_alexnet():
    try:
        image = request.files['file'].read()
        prediction = predict(models_dict['alexnet'], image)
        return jsonify({'message': 'Prediction using AlexNet', 'prediction': prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info("Listening on port 5000...")
    app.run(host='0.0.0.0', port=5000)
