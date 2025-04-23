#Neural network by Salvatore Naro
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.rete = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        tensore = torch.tensor([[1., 2., 3., 4., 5.],
                                [6., 7., 8., 9., 10.]], device=device)
        tensorey = torch.tensor([[11., 12., 13., 14., 15.],
                                 [16., 17., 18., 19., 20.]], device=device)
        f = tensorey * tensore
        x_strided = torch.as_strided(tensore, (2, 4), (5, 1)) 
        k = torch.as_strided(f, (2, 4), (5, 1)) 

        tensore[:, -1] *= 2 
        ones = torch.ones([2, 3, 4], device=device)
        zeros = torch.zeros([2, 3, 4], device=device)
        randx = torch.rand([2, 3, 4], device=device)
        randy = torch.rand([4, 2, 3, 1], device=device)
        randtot = torch.sin(randx) * torch.sin(randy)

        
        print("randx shape:", randx.shape)
        print("randx reshaped:", randx.view(1, 24))
        print("randy stride:", randy.stride())
        print("randtot stride:", randtot.stride())
        print("zeros shape:", zeros.shape)
        print("ones shape:", ones.shape)
        print("ones unsqueezed * 3:", torch.unsqueeze(ones, dim=1) * 3)
        print("tensore reshaped:", tensore.reshape([1, 10]))
        print("k (strided f):", k)
        print("k stride:", k.stride())
        print("x_strided:", x_strided)

        
        return self.rete(x_strided)


def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = Network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data = data[:, :4] 
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)  
            dummy_target = torch.randn(output.shape, device=device)  
            loss = criterion(output, dummy_target)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    model = Network()
    input_tensor = torch.randn(2, 4)  
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(output)
    train_model()
