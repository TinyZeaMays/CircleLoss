import os

import torch
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from circle_loss import CircleLossBackward, convert_label_to_similarity


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

    def forward(self, inp: Tensor) -> Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return nn.functional.normalize(feature)


def main(resume: bool = True) -> None:
    model = Model()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loader = get_loader(is_train=True, batch_size=64)
    val_loader = get_loader(is_train=False, batch_size=2)
    loss_backward = CircleLossBackward(m=0.25, gamma=256)

    if resume and os.path.exists("resume.state"):
        model.load_state_dict(torch.load("resume.state"))
    else:
        for epoch in range(20):
            for img, label in tqdm(train_loader):
                model.zero_grad()
                pred = model(img)
                loss_backward(*convert_label_to_similarity(pred, label))
                optimizer.step()
        torch.save(model.state_dict(), "resume.state")

    tp = 0
    fn = 0
    fp = 0
    thresh = 0.8
    for img, label in val_loader:
        pred = model(img)
        gt_label = label[0] == label[1]
        pred_label = torch.sum(pred[0] * pred[1]) > thresh
        if gt_label and pred_label:
            tp += 1
        elif gt_label and not pred_label:
            fn += 1
        elif not gt_label and pred_label:
            fp += 1

    print("Recall: {:.4f}".format(tp / (tp + fn)))
    print("Precision: {:.4f}".format(tp / (tp + fp)))


if __name__ == "__main__":
    main()
