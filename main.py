from flask import Flask, jsonify, request
import torch
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Define the transform to preprocess images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the API endpoint for predicting image classes
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image'].read()

    # Preprocess the image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
    predictions = torch.nn.functional.softmax(output[0], dim=0)

    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)

