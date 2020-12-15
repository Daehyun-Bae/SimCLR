import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model
from resnet50 import Resnet50


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.lb_seq, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target, _ in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
            #                          .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--server', action='store_true', help='Run in the server')
    parser.add_argument('--data', dest='data', default='market', type=str, choices=['market', 'duke'])
    parser.add_argument('--exp', dest='exp', default='201211', type=str)

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs, exp = args.batch_size, args.epochs, args.exp
    IS_SERVER = args.server
    DATA = args.data
    model_pth = 'results/201211_market_128_0.5_50_512_500_model.pth'

    # data prepare
    # train_root = utils.get_data_root(data=DATA, server=IS_SERVER, target='train')
    # train_dataset = utils.ReidDataset(data_path=train_root)
    # # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    #                           drop_last=True)

    memory_root = utils.get_data_root(data=DATA, server=IS_SERVER, target='test')
    memory_dataset = utils.ReidDataset(data_path=memory_root)
    memory_loader = DataLoader(memory_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_root = utils.get_data_root(data=DATA, server=IS_SERVER, target='test')
    test_dataset = utils.ReidDataset(data_path=test_root, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # model setup and optimizer config
    # model = Model(feature_dim).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('NUM DEVICE: {} DEVICE: {}'.format(torch.cuda.device_count(), device))
    model = Resnet50(pretrained=False, max_pool=False, feature_dim=feature_dim).cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    else:
        model = torch.nn.DataParallel(model, device_ids=[0])
    # model.to(device)
    load_state = torch.load(model_pth)
    model.load_state_dict(torch.load(model_pth))

    c = len(memory_dataset.lb_ids_uniq)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(exp, feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0

    test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)

    print("TEST ACC@1: {:.4f} ACC@5: {:.4f}".format(test_acc_1, test_acc_5))