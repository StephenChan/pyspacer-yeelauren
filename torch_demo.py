import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained EfficientNet-B1 model
model = EfficientNet.from_pretrained('efficientnet-b1')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the input image
image = Image.open('path/to/your/image.jpg')
image = transform(image).unsqueeze(0)
image = image.to(device)

# Forward pass through the model
with torch.no_grad():
    output = model(image)

# Get the predicted class probabilities
probabilities = torch.nn.functional.softmax(output, dim=1)[0]

# Print the top 5 predicted classes and their probabilities
top5_prob, top5_classes = torch.topk(probabilities, k=5)
for prob, class_idx in zip(top5_prob, top5_classes):
    print(f'Class: {class_idx.item()}, Probability: {prob.item()}')
