import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from collections import defaultdict


# Dataset for re-ID dataset

normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet - official


class ReidDataset(Dataset):
    def __init__(self, data_path, is_train=True, *args, **kwargs):
        super(ReidDataset, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.n_cams = len(set(self.lb_cams))

        # Camera style augmentation
        self.n_imgs = len(self.imgs) // self.n_cams

        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        if is_train:
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop((256, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(normalize[0], normalize[1]),
                transforms.RandomErasing(p=0.5)
            ])

            # self.trans = transforms.Compose([
            #     # transforms.RandomResizedCrop(size=(288, 144), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            #     transforms.Resize((288, 144)),
            #     transforms.RandomCrop((256, 128)),
            #     # transforms.Grayscale(num_output_channels=3),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(normalize[0], normalize[1]),
            #     transforms.RandomErasing(p=0.5)
            # ])

        else:
            self.trans = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(normalize[0], normalize[1]),
            ])

            # self.trans_tuple = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(normalize[0], normalize[1])
            # ])
            # self.Lambda = transforms.Lambda(
            #     lambda crops: [self.trans_tuple(crop) for crop in crops])
            # if test_augmentation:
            #     self.trans = transforms.Compose([
            #         transforms.Resize((288, 144)),
            #         transforms.TenCrop((256, 128)),
            #         self.Lambda,
            #     ])
            # else:
            #     self.trans = transforms.Compose([
            #         transforms.Resize((256, 128)),
            #         # transforms.RandomResizedCrop(size=(288, 144), scale=(0.8, 1.0)),
            #         # transforms.RandomCrop((288, 144)),
            #         transforms.ToTensor(),
            #         transforms.Normalize(normalize[0], normalize[1]),
            #     ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

        self.lb_seq = []
        lb_dict = defaultdict()
        sorted_uniq_id = sorted(self.lb_ids_uniq)
        for lb, item in enumerate(sorted_uniq_id):
            lb_dict[item] = lb
        for i in self.lb_ids:
            self.lb_seq.append(lb_dict[i])

        # print('NUM Class: ', max(self.lb_seq) + 1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            img_org = Image.open(self.imgs[idx])

            pos_1 = self.trans(img_org)
            pos_2 = self.trans(img_org)

            return pos_1, pos_2, self.lb_seq[idx], self.imgs[idx]
        except TypeError:
            print('ERROR idx: ', idx)
            exit()


def get_data_root(data='market', server=True, target='train', stargan=False):
    if target == 'train':
        dir_name = 'bounding_box_train'
    elif target == 'test':
        dir_name = 'bounding_box_test'
    elif target == 'query':
        dir_name = 'query'
    if stargan:
        dir_name = 'bounding_box_train_camstyle_stargan4reid'

    if data == 'market':
        return '/datasets/reid/Market-1501-v15.09.15/' + dir_name if server else 'D:\\datasets\\re_id\\Market-1501-v15.09.15\\' + dir_name
    elif data == 'duke':
        return '/datasets/reid/DukeMTMC-reID/' + dir_name if server else 'D:\\datasets\\re_id\\DukeMTMC-reID\\' + dir_name

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

if __name__ == '__main__':
    data_root = get_data_root(data='market', server=False)
    ds = ReidDataset(data_path=data_root, is_train=True)
    print(len(ds))
    # from utils.batch_sampler import BatchSampler
    # sampler = BatchSampler(dataset=ds, n_class=5, n_num=3)
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, num_workers=4)
    # dl = DataLoader(dataset=ds, batch_sampler=sampler, num_workers=4)
    diter = iter(dl)
    # for i in range(len(ds)):
    #     print(f'{ds.imgs[i]}\t{ds.lb_ids[i]}\t{ds.lb_seq[i]}')
    ds_dict = ds.lb_img_dict
    print(ds.n_imgs)

    print(ds.lb_ids == sorted(ds.lb_ids))
    # for _ in range(5):
    #     img, img2, lb, fn = next(diter)
    #     for i in range(4):
    #         print(f'{fn[i]} --> {lb[i]} {img[i].shape}')
    #     print('-'*60)