import torch
from torch import nn, Tensor


class NormLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(NormLinear, self).__init__(in_features, out_features, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return nn.functional.linear(nn.functional.normalize(inp),
                                    nn.functional.normalize(self.weight))


class CircleLossLikeCE(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLossLikeCE, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp: Tensor, label: Tensor) -> Tensor:
        a = torch.clamp_min(inp + self.m, min=0).detach()
        src = torch.clamp_min(
            - inp.gather(dim=1, index=label.unsqueeze(1)) + 1 + self.m,
            min=0,
        ).detach()
        a.scatter_(dim=1, index=label.unsqueeze(1), src=src)

        sigma = torch.ones_like(inp, device=inp.device, dtype=inp.dtype) * self.m
        src = torch.ones_like(label.unsqueeze(1), dtype=inp.dtype, device=inp.device) - self.m
        sigma.scatter_(dim=1, index=label.unsqueeze(1), src=src)

        return self.loss(a * (inp - sigma) * self.gamma, label)


class CircleLossBackward(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLossBackward, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        sigma_p = 1 - self.m
        sigma_n = self.m

        logit_p = ap * (sp - sigma_p) * self.gamma
        logit_n = an * (sn - sigma_n) * self.gamma

        loss = torch.log(1 + torch.clamp_max(torch.exp(logit_n).sum() * torch.exp(- logit_p).sum(), max=1e38))
        z = - torch.exp(- loss) + 1

        sp.backward(gradient=z * (- ap) * torch.softmax(logit_p, dim=0) * self.gamma)
        sn.backward(gradient=z * an * torch.softmax(logit_n, dim=0) * self.gamma)

        return loss.detach()


if __name__ == "__main__":
    inp_sp = torch.rand(100, requires_grad=True)
    inp_sn = torch.rand(200, requires_grad=True)

    circle_loss_backward = CircleLossBackward(m=0.25, gamma=256)
    circle_loss = circle_loss_backward(inp_sp, inp_sn)

    print(circle_loss)
