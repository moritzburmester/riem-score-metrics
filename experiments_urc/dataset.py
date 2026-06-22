from dataloader import AlphaNum, AlphaNumV2, ImageLoader
from torchvision import datasets, transforms
import os
import argparse

## code adapted from https://github.com/VictorBoutin/RiemannEBM/blob/main/utils/dataset.py

def get_dataloader(data_root, dataset, image_size=128, sequential=False, letter=None, rot_dist='uniform'):
    path_to_data = os.path.join(data_root, dataset)
    if dataset == 'alphanum':
        #dloader = AlphaNum(data_root=path_to_data,
        #                   sequential=sequential,
        #                   one_letter=letter)
        dloader = AlphaNumV2(data_root=path_to_data,
                           sequential=sequential,
                           one_letter=letter,
                           rot_dist=rot_dist)
    elif dataset =='cifar10':
        transf = transforms.ToTensor()
        dloader = datasets.CIFAR10(path_to_data, train=True, download=False, transform=transf)


    return dloader

def get_dataloader_im(data_root,
                      dataset,
                      image_size=128,
                      device='cpu',
                      ae_name="stable_diff_14",
                      ambiant=False):
    path_to_data = os.path.join(data_root, dataset)
    if dataset == 'afhq' or dataset == 'celebahq':
        dloader = ImageLoader(data_root=path_to_data,
                              image_size=image_size,
                              device=device,
                              ae_name=ae_name,
                              load_ambiant=ambiant)
    elif dataset == 'cifar10':
        transf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dloader = datasets.CIFAR10(path_to_data, train=True, download=True, transform=transf)

    else:
        raise NotImplementedError()

    return dloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocessing data")
    ## DATA args
    parser.add_argument('--dataset', type=str, default='afhq', choices=['alphanum', 'cifar10', 'afhq', 'celebahq'])
    parser.add_argument("--data_root", type=str, default="/media/data_cifs_lrs/projects/prj_mental/datasets")



    args = parser.parse_args()


    get_dataloader_im(data_root=args.data_root,
                      dataset=args.dataset,
                      image_size=128,
                      device="cuda:0",
                      ae_name="stable_diff_14_aug",
                      ambiant=True
                      )