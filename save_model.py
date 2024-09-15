from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Define the model architectures
def load_model(model_name):
    try:
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=False, num_classes=3)
            model.load_state_dict(torch.load("models/densenet121.pth", map_location='cpu'))
        elif model_name == 'resnet':
            model = models.resnet50(pretrained=False, num_classes=3)
            model.load_state_dict(torch.load("models/resnet50.pth", map_location='cpu'))
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=False, num_classes=3)
            model.load_state_dict(torch.load("models/alexnet.pth", map_location='cpu'))
        else:
            raise ValueError("Invalid model name")
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

# Load models
try:
    model_densenet = load_model('densenet121')
    model_resnet = load_model('resnet')
    model_alexnet = load_model('alexnet')
except RuntimeError as e:
    print(f"Model loading error: {e}")
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
        raise RuntimeError(f"Error processing image: {str(e)}")

# Define prediction route
def predict(model, image_bytes):
    try:
        input_data = preprocess_image(image_bytes)
        with torch.no_grad():
            prediction = model(input_data)
        prediction = torch.softmax(prediction, dim=1).tolist()
        return prediction
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

# Route for inference using DenseNet121
@app.route('/predict/densenet', methods=['POST'])
def predict_densenet():
    try:
        image = request.files['file'].read()
        prediction = predict(model_densenet, image)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for inference using ResNet
@app.route('/predict/resnet', methods=['POST'])
def predict_resnet():
    try:
        image = request.files['file'].read()
        prediction = predict(model_resnet, image)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for inference using AlexNet
@app.route('/predict/alexnet', methods=['POST'])
def predict_alexnet():
    try:
        image = request.files['file'].read()
        prediction = predict(model_alexnet, image)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Listening on port 5000...")
    app.run(host='0.0.0.0', port=5000)
