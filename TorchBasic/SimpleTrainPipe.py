"""
1 define model
2 construct loss
3 training loop
 - forward pass
 - backward pass
 - update weight
"""

import torch
import torch.nn as nn

# default liniear model weight float32
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([[5]],dtype=torch.float32)

n_samples, n_features = X.shape
lr = 0.01
n_inters = 100

input_size = n_features
output_size = n_features

#  Way 1 default model
# model = nn.Linear(input_size, output_size)

# Way 2or custom model
class LRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LRegression, self).__init__()
        self.l1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.l1(x)

model = LRegression(input_size, output_size)


print(f"predicting: x= 5: {model(X_test).item():.3f}")

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

for epoch in range(n_inters):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f"{w[0]},{b}")
        print(f"epoch: {epoch}, loss: {l:.8f}")

print(f"predicting: x= 5: {model(X_test).item():.3f}")