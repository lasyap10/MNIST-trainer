import torch
import torch.nn as nn
from dataloader import building_dataloaders
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from cnn import Net
from dataset import data

torch.manual_seed(1) #reduces randomness

def train(model, training_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (image, label) in enumerate(training_loader):
        image, label = image.to(device), label.to(device) #moving batch with labels to the GPU
        optimizer.zero_grad() #clears the gradients each time for a fresh, accurate start
        output = model(image) #send batch through forward pass
        loss = F.nll_loss(output, label) #returns one number for how wrong the model is
        loss.backward() #computes the gradients for every weight in batch
        optimizer.step() #uses gradients to nudge weights towards bottom
        if batch_idx % 100 == 0:
            print(f"epoch: {epoch} \nbatch number: {batch_idx} \nloss: {loss.item()}")

def test(model, device, test_loader):
    #starts off one forward pass per test batch (10 test batches of 1000 images each)
    model.eval() #dropout is turned off - evalutated on every image without random zeroing out
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = F.nll_loss(output, label, reduction = 'sum').item() #to sum the overall loss of the batch
            test_loss += loss #this is the loss total from testing a batch
            prediction = output.argmax(dim = 1) #finds the maximum activation of the value and returns the index of that activation per image
            track_correct_pred = prediction == label #comparing the predictions to actual correct values
            correct += track_correct_pred.sum().item() #gives count of truth values
    average_loss = test_loss / len(test_loader.dataset)
    accuracy = (correct / len(test_loader.dataset)) * 100

    print(f"average loss: {average_loss} \naccuracy: {accuracy}")


def main():
    #take contents of train and test and run it in the right order
    EPOCH = 5
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    training_loader, test_loader = building_dataloaders()
    #creating model and moving it to the device
    model = Net().to(device)
    #creating optimizer
    optimizer = optim.Adadelta(model.parameters(), lr = 1)
    #creating scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    for epoch in range(1, EPOCH + 1):
        train(model, training_loader, optimizer, device, epoch)
        test(model, device, test_loader)
        scheduler.step() #fixes the learning rate, turns down the step size for the next epoch for the optimizer to use
        #stabilizes the model as lr decreases little by little, allowing for optimizer to move the loss to a lower and lower point, eventually stabilizing
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("model saved")

if __name__ == "__main__":
    main()















        

