import torch
# import torchvision
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from sklearn.datasets import load_wine
from torch.nn.functional import normalize

class WineData(Dataset):
    def __init__(self, transform=None):
        X,y=load_wine(return_X_y=True)
        #X,y = X.data, y.data
        self.X = torch.from_numpy(X.astype(np.float32))
        # self.X= normalize(self.X , p=3.0, dim = 0)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.transform = transform
        self.n_feature = self.X.shape[0]
    
    def __getitem__(self, index):
        sample = self.X[index],self.y[index]
        if self.transform:
            self.transform((self.X,self.y))
        return sample
    
    def __len__(self):
        return self.n_feature
    
class MulTransform:
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineData()

train_loader = DataLoader(dataset=dataset, batch_size=178)
# iterator=iter(loader)

# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, 100) 
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(100, num_classes)  


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=13, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (applies Softmax)
num_epochs = 50_000
optimizer = torch.optim.SGD(model.parameters(), lr=2e-4) 
loss = 0
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):  
        #print(labels)

        # Forward pass
        outputs = model(features)
        #print(f"prediction: {torch.max(outputs,1)[1]}")
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    if (epoch) % 100 == 0:
        print (f'Epoch [{epoch}/{num_epochs}],  Loss: {loss.item():.12f}')
