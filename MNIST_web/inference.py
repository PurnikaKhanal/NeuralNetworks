import torch
import base64
import numpy as np
from PIL import Image
import io
from model import MNISTNet
from torchvision import transforms

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = MNISTNet()
model.load_state_dict(
    torch.load("MNIST_CNN.pth", map_location=device, weights_only=False)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_base64_image(image_base64: str):

    # remove metadata if present
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(img_data)).convert("L")

    # Ensure the image is already 28x28
    if image.size != (28, 28):
        raise ValueError("Input image must be 28x28 pixels.")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction