import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from backend.core.NeuralNetwork import NeuralNetwork

def estimate(image_bytes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor()])

    model = NeuralNetwork().to(device)
    model.load_params()
    model.eval()

    image = Image.open(BytesIO(image_bytes))
    image = image.convert('L').resize((28, 28))
    image = transform(image).unsqueeze(0)

    output = model(image.to(device))
    pred = output.data.max(1, keepdim=True)[1]

    return pred.item()
