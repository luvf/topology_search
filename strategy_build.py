import torch
import torch.nn as nn
from torch import Tensor


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # parametrize
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, args.n_features),
        )

    def forward(self, x):
        out = self.main(x)
        return out


adversarial_loss = torch.nn.BCELoss()
classification_loss =  torch.nn.MSELoss()


def loss_batch(model, x, y, opt=None):
    clf = model(x)

    loss = classification_loss(clf, y)

    # center loss more tricky

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


def fit(epochs, model, opt, dataset):
    for epoch in range(epochs):
        model.train()
        for x, y in dataset:
            loss = loss_batch(model, x, y, opt)
        print(epoch, loss)

        # TODO
        # model.eval()
        # with torch.no_grad():
        #    losses, nums = zip(
        #        *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        #    )
        # val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # print(epoch, val_loss)



def buid_strategy(env, policy):
    """
    :param env: pypownet environement
    :param policy:
    :return:
    """
