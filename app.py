#to predict an digit for an image
import torch
from cnn import Net
from dataset import data

#cpu(does work one after the other) - general purpose proccessor
#gpu - specialized processor(good for parallel processing) --> can handle more complex math operations than cpu which
#is why we run the model on the gpu an then allow for the model to predict on the cpu

model = Net()
weights = torch.load("mnist_cnn.pt", map_location="cpu")
model.load_state_dict(state_dict=weights)
model.eval()
train, test = data()
#grabbing one test image
image, label = test[0]
image = image.unsqueeze(0) #creates fourth dimention (a batch dimension)

print(image.shape, label)


with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()
    print(f"prediction: {prediction} \ntrue label: {label}")










