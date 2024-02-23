# make necessary imports
from torch import functional
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torch_lr_finder import *
import torch.optim as optim
import matplotlib.pyplot as plt



# %%
# transform class to make images sharper
class Sharpie(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        enhancer = ImageEnhance.Sharpness(x)
        img = enhancer.enhance(self.factor)
        return img


# %%
torch.manual_seed(42)
transformations = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(3),
    transforms.Lambda(lambda x: torchvision.transforms.functional.adjust_contrast(x, 2)),
    transforms.Lambda(lambda x: torchvision.transforms.functional.adjust_contrast(x, 2)),
    transforms.Lambda(lambda x: torchvision.transforms.functional.adjust_gamma(x, 2)),
    Sharpie(factor=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# %%
device = 'cpu'
# %%
dataset = datasets.ImageFolder('./manga_expressions', transform=transformations)
# %%
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train, val, test = data.random_split(dataset=dataset, lengths=[train_size, val_size, test_size],
                                      generator=torch.Generator().manual_seed(42))

# %%
train_loader = data.DataLoader(train, shuffle=True, batch_size=15)
test_loader = data.DataLoader(test, shuffle=True, batch_size=10)


# %%
# show some images
def imshow(img):
    ax = plt.subplots(figsize=(30, 90))
    inp = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, interpolation='nearest')


# %%

inputs, classes = next(iter(train_loader))
grid = torchvision.utils.make_grid(inputs[:7])
unique_classes = np.unique(classes[:9].numpy())
print(np.array(dataset.classes)[unique_classes])
imshow(grid)


# %%
def train_model(epoch, train_loader, optimizer, model):
    model.train()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# %%
def test_model(epoch, test_loader, model):
    model.eval()
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            running_loss += criterion(output, target).item() * inputs.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            running_acc += pred.eq(target.view_as(pred)).sum().item()

    running_loss = running_loss / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_loss, running_acc, len(test_loader.dataset),
        100. * running_acc / len(test_loader.dataset)))


# %%
criterion = nn.CrossEntropyLoss()
my_model = models.resnet50(pretrained=True)

for param in my_model.parameters():
    param.requires_grad = False

in_feats = my_model.fc.in_features
my_model.fc = nn.Linear(in_feats, 7)
my_model = my_model.to(device)
# %%
optimizer = optim.Adam(my_model.parameters(), lr=0.0010974987654930562, weight_decay=0.00000001)
# %%
# training only last layer
""
for epoch in range(1, 6):
    print('\nTraining:')
    train_model(epoch, train_loader, optimizer, my_model)
    print('\nTesting:')
    test_model(epoch, test_loader, my_model)
# %%
for param in my_model.parameters():
    param.requires_grad = True

# training full model
for epoch in range(1, 11):
    print('\nTraining:')
    train_model(epoch, train_loader, optimizer, my_model)
    print('\n Testing:')
    test_model(epoch, test_loader, my_model)
# Save the model
torch.save(my_model.state_dict(), 'my_model.pth')
#comment out for tuning """