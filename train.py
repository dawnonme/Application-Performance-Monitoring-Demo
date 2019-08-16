import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import APMNetwork
from dataset import get_dataloader
from mock_data import KPIS

cfg = {
    'name': 'exp1',
    'in_dim': len(KPIS),
    'out_dim': 4,
    'lr': 0.005,
    'max_epochs': 1,
    'batch_size': 32,
    'cuda': False,
    'save_interval': 1,
    'save_path': './checkpoints',
    'eval': False,
    'pretrained_model_path': None,
    'start_epoch': 1,
}

use_cuda = cfg['cuda'] and torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

if not os.path.exists(cfg['save_path']):
    os.mkdir(cfg['save_path'])

checkpoint_path = cfg['save_path'] + '/' + cfg['name']
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

model = APMNetwork(cfg['in_dim'], cfg['out_dim'])

if cfg['pretrained_model_path']:
    model.load_state_dict(torch.load(cfg['pretrained_model_path']))

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

train_loader = get_dataloader('train', cfg['batch_size'])
test_loader = get_dataloader('test', cfg['batch_size'])


def train():
    model.train()

    print('Start training...')
    for epoch in range(cfg['start_epoch'] - 1, cfg['max_epochs']):
        running_loss = 0
        for idx, (inputs, target) in enumerate(train_loader, 0):
            optimizer.zero_grad()

            inputs = inputs.float()

            inputs.to(device)
            target.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, target.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (idx + 1) % 100 == 0:
                print('Epoch: %d | Iter: %d | LR: %f | Loss: %f' %
                      (epoch + 1, idx + 1, cfg['lr'], running_loss))
                running_loss = 0

        if (epoch + 1) % cfg['save_interval'] == 0:
            torch.save(model.state_dict(), cfg['save_path'])


def test():
    model.eval()
    print('Start testing...')
    num_sample = num_correct = 0
    with torch.no_grad():
        for idx, (inputs, target) in enumerate(test_loader, 0):
            inputs.to(device)
            target.to(device)
            target = target.long()
            outputs = model(inputs.float())

            _, ans = torch.max(outputs, 1)
            num_sample += target.shape[0]
            num_correct += (ans == target).sum().item()

            print('Iter: %d | Num Samples Now: %d | Num Correction Now: %d' %
                  (idx, num_sample, num_correct))

        print('Accuracy: %.2f/1.00' % (num_correct / num_sample))


if __name__ == '__main__':
    if not cfg['eval']:
        train()

    test()
