from torchvision import datasets, transforms

"""
psuedocode:
1. start off getting mnist
2. then transform the mnist dataset into tensors
3. then normalize the tensors
"""

# storing transformations that should be done on grayscale image
# first converts image pixels to tensor also adding a channel, then normalizes 
# it for more stabilized training
def transformation():
    transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize((0.1307,),(0.3081,))]) #average brightness and how spread out the values are from that average 
    return transform

# applying the transforms to the testing and training image datasets
def data():
    transforming = transformation()
    training_data = datasets.MNIST("data", train = True, download = True, transform = transforming)
    testing_data = datasets.MNIST("data", train = False, download = True, transform = transforming)
    return training_data, testing_data


if __name__ == "__main__":
    train, test = data()
    image, label = train[0]
    print(f"image: {image.shape} \nlabel: {label}")
    print(f"min: {image.min().item()}")
    print(f"max: {image.max().item()}")
    print(f"train length: {len(train)}, test length: {len(test)}")