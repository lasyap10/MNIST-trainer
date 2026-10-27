from dataset import data
from torch.utils.data import DataLoader

"""
1. Build dataloader function
2. In the function, bring in the datasets from dataset.py and unpack them into variables
3. Create two loader variables in a new function and return them
4. set batch size and shuffle for training and testing loaders
"""

def building_dataloaders():
    training, testing = data()
    training_loader = DataLoader(training, batch_size = 64, shuffle = True)
    testing_loader = DataLoader(testing, batch_size = 1000, shuffle = False)
    return training_loader, testing_loader


if __name__ == "__main__":
    training_loader, testing_loader = building_dataloaders()
    #pulling out one batch for checking shape
    i = iter(training_loader)
    image, label = next(i)
    iterator = iter(testing_loader)
    image2, label2 = next(iterator)

    print(f"Number of batches in train set: {len(training_loader)}")
    print(f"Number of batches in test set: {len(testing_loader)}")

    #printing shape and labels in batch #1
    print(f"Train image dimensions: {image.shape}")
    print(f"Train labels: {label.shape}")
    print(f"Test image dimensions: {image2.shape}")
    print(f"Test labels: {label2.shape}")







