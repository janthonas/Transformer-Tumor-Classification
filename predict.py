from network_architecture import transform_pipeline, transfer_model

from PIL import Image
import torch


# Loads the best available models
state_dict = torch.load("models/best_tumor_model.pth")
transfer_model.load_state_dict(state_dict)

labels = ['Healthy', 'Tumor']

# Selects a given file
filename = 'mri-tumor-org/test/Tumor/glioma (127).jpg'

# Applies the transform pipeline to the image
img = Image.open(filename)
img = transform_pipeline(img)
img = img.unsqueeze(0)

# Makes a prediction on the image
prediction = transfer_model(img)
prediction = prediction.argmax()
print(labels[prediction])