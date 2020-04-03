from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from circle_loss import CircleLossLikeCE, NormLinear


def get_loader(is_train: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=MNIST(root="./data", train=is_train, transform=ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=is_train,
    )


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.linear = NormLinear(32, 10)

    def forward(self, inp: Tensor) -> Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return self.linear(feature)


def main():
    model = Model()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = get_loader(is_train=True, batch_size=64)
    val_loader = get_loader(is_train=False, batch_size=64)
    criterion = CircleLossLikeCE(m=0.25, gamma=30)

    for img, label in tqdm(train_loader):
        model.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

    top = 0
    bot = 0
    for img, label in val_loader:
        pred = model(img)
        result = label.eq(pred.max(dim=1)[1])
        top += float(result.sum())
        bot += float(result.numel())

    print("Accuracy: {:.4f}".format(top / bot))


if __name__ == "__main__":
    print("CircleLossLikeCE is deprecated. Use CircleLossBackward instead.")
