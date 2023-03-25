import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json

# Define the transformation to be applied to input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the pre-trained ResNet50 model
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.eval()

def predict(request):
    # Parse the request body to extract the image data
    body = request.get_data()
    img = Image.open(io.BytesIO(body))

    # Apply the transformation to the input image
    img_t = transform(img)
    img_t = torch.unsqueeze(img_t, 0)

    # Make a prediction with the pre-trained ResNet50 model
    with torch.no_grad():
        output = model(img_t)
        prediction = torch.argmax(output).item()

    # Return the prediction as a JSON response
    response = {'prediction': prediction}
    return json.dumps(response)
