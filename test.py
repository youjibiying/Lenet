from __future__ import print_function

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from CNN_mnist import Lenet


def load_model(device):
    model = Lenet(num_features=4, num_features1=2,act_type='lrelu')
    model.load_state_dict(torch.load('./model/Lenet_nums_features4_2_lrelu_params.pkl'))
    model.to(device)
    return model


def test(model, device, test_loader, accur):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss ,
            # reduction default is "mean"https://pytorch.org/docs/stable/nn.functional.html#nll-loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accur.append(correct / len(test_loader.dataset))


def main():
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_batch_size = 1000
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    accur = []
    epochs = 3
    for i in range(epochs):
        test(model, device, test_loader, accur)
    plt.plot(range(1, 1 + epochs), accur)
    plt.show()


if __name__ == '__main__':
    main()
