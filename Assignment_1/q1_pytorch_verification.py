import json, torch, torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2); self.layer2 = nn.Linear(2, 2)
        self.layer3 = nn.Linear(2, 2); self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.layer1(x)); x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.layer3(x)); return self.sigmoid(self.output(x))

model = SimpleNet().double()
with torch.no_grad():
    model.layer1.weight.copy_(torch.tensor([[0.1, 0.3], [0.2, 0.4]], dtype=torch.float64))
    model.layer1.bias.copy_(torch.tensor([0.1, 0.1], dtype=torch.float64))
    model.layer2.weight.copy_(torch.tensor([[0.3, 0.5], [0.4, 0.2]], dtype=torch.float64))
    model.layer2.bias.copy_(torch.tensor([0.1, 0.1], dtype=torch.float64))
    model.layer3.weight.copy_(torch.tensor([[0.2, 0.4], [0.3, 0.1]], dtype=torch.float64))
    model.layer3.bias.copy_(torch.tensor([0.1, 0.1], dtype=torch.float64))
    model.output.weight.copy_(torch.tensor([[0.5, 0.3]], dtype=torch.float64))
    model.output.bias.copy_(torch.tensor([0.1], dtype=torch.float64))

x = torch.tensor([0.5, 0.8], dtype=torch.float64)
y = torch.tensor([1.0], dtype=torch.float64)
y_pred = model(x)
loss = 0.5 * (y_pred - y) ** 2
loss.backward()


print(f"y_pred = {y_pred.item():.6f}   loss = {loss.item():.6f}")
for name, p in model.named_parameters():
    print(name, p.grad.numpy().round(6))
