import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
from manga_expression_classification_model import my_model, train_loader, train_model, test_loader, test_model

# Define the hyperparameters
lr = 0.001
weight_decay = 0.001

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

# Find the optimal learning rate
lr_finder = LRFinder(my_model, optimizer, criterion, device="cpu")
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()
plt.show()
lr = lr_finder.history["lr"][lr_finder.history["loss"].index(lr_finder.best_loss)]
print(f"Optimal learning rate value: {lr}")

# Train the model with the optimal hyperparameters
for epoch in range(1, 10):
    train_model(epoch, train_loader, optimizer, my_model)
    test_model(epoch, test_loader, my_model)
