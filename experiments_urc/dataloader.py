import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset, ImageFolder
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import warnings
import os
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
from PIL import Image
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import Resize, CenterCrop
from torch.utils.data import DataLoader
from monitoring import plot_img
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
import random
import torch.distributions as dist

import random
# import torchvision.transforms.InterpolationMode as Interp
from torchvision.transforms import InterpolationMode as Interp


## code from https://github.com/VictorBoutin/RiemannEBM/blob/main/utils/dataloader.py


## transform_name
hflip = torchvision.transforms.functional.hflip
res_cro = torchvision.transforms.functional.resized_crop

mapping_AE_file = {"stable_diff_14": "afhq_sd_14_latent.pt",
                   "stable_diff_14_aug": "afhq_sd_14_latent_aug.pt",
                   "stable_diff_14_aug_max": "afhq_sd_14_latent_aug_max.pt",
                   "stable_diff_14_aug_max_max": "afhq_sd_14_latent_aug_max_max.pt",
                   "stable_diff_14_aug_max_rot": "afhq_sd_14_latent_aug_max_rot.pt",
                   "stable_diff_14_celeba": "celeba_sd_14_latent.pt"
                   }

crop = torchvision.transforms.RandomResizedCrop(
    128, scale=[0.8, 1.0], ratio=[0.9, 1.1])
rand_crop = torchvision.transforms.Lambda(
    lambda x: crop(x) if random.random() < 1 else x)
tr_max = transforms.Compose([
    rand_crop,
    torchvision.transforms.RandomHorizontalFlip(),
])

crop_2 = torchvision.transforms.RandomResizedCrop(
    128, scale=[0.7, 1], ratio=[0.8, 1.2])
rand_crop_2 = torchvision.transforms.Lambda(
    lambda x: crop_2(x) if random.random() < 0.5 else x)
tr_max_2 = transforms.Compose([
    rand_crop_2,
    torchvision.transforms.RandomHorizontalFlip(),
])

rot = transforms.Compose([
    torchvision.transforms.RandomRotation((-10, 10)),
    torchvision.transforms.CenterCrop((112, 112)),
    torchvision.transforms.Resize((128, 128))
])


def custom_tr(x):
    rand_nb = random.random()
    if 0 <= rand_nb < 0.33:
        return crop_2(x)
    elif 0.33 <= rand_nb < 0.66:
        return rot(x)
    else:
        return x


rand_rot_and_crop = torchvision.transforms.Lambda(
    lambda x: custom_tr(x))

tr_rot = transforms.Compose([
    rand_rot_and_crop,
    torchvision.transforms.RandomHorizontalFlip(),
])


class ImageLoader(VisionDataset):
    def __init__(self,
                 data_root,
                 image_size,
                 device='cuda:0',
                 ae_name="stable_diff_14",
                 load_ambiant=False):
        super().__init__()
        self.data_path = data_root
        self.image_size = image_size
        self.ae_name = ae_name
        self.device = device

        self.process = VaeImageProcessor(do_convert_rgb=True)
        if ae_name == 'stable_diff_14_aug_max_max' or ae_name == 'stable_diff_14_aug_max_rot':
            self.nb_aug = 200
        else:
            self.nb_aug = 100
        if 'stable_diff_14' in ae_name:
            # if ae_name == 'stable_diff_14' or ae_name == 'stable_diff_14_aug' or ae_name == 'stable_diff_14_aug_max' or ae_name:
            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True
            )
            self.vae.enable_tiling()
            self.vae.enable_slicing()

        self.latent_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        self.train_latent_save_path = os.path.join(self.data_path, f"train_{mapping_AE_file[self.ae_name]}")

        if load_ambiant:
            self.train_ambiant_save_path = os.path.join(self.data_path, f"train_ambiant_afhq")

        if os.path.exists(self.train_latent_save_path):
            self.data_train = torch.load(self.train_latent_save_path, weights_only=False)
        else:
            # self.base_train_dataset = self.image_base_dataset(
            #    "train", self.latent_transform
            # )
            if self.ae_name == 'stable_diff_14_celeba':
                self.base_train_dataset = self.image_base_dataset(
                    "celeba", self.latent_transform
                )
            else:
                self.base_train_dataset = self.image_base_dataset(
                    "total", self.latent_transform
                )
            self.data_train = self._get_latent(database=self.base_train_dataset,
                                               save_path=self.train_latent_save_path)

        if load_ambiant:
            self.base_train_ambiant_dataset = self.image_base_dataset(
                "total", self.latent_transform
            )

            self.data_train_ambiant = self._get_ambiant(database=self.base_train_ambiant_dataset,
                                                        save_path=self.train_ambiant_save_path)

    def to_im(self, z):
        with torch.no_grad():
            z = z * self.data_train['std'] + self.data_train['mean']
            x = self.vae.to(z.device).decode(z)
        return x.sample

    def image_base_dataset(self, split, transform):
        if split == "train":
            path = os.path.join(self.data_path, "train")
        elif split == "total":
            path = os.path.join(self.data_path, "total")
        elif split == "celeba":
            if os.path.exists(os.path.join(self.data_path, "CelebAMask-HQ")):
                path = os.path.join(self.data_path, "CelebAMask-HQ", "CelebA-HQ-img")
            elif os.path.exists(self.data_path):
                path = self.data_path
            else:
                raise FileNotFoundError("CelebA dataset not found in the specified path.")
        elif split == "val":
            path = os.path.join(self.data_path, "val")
        else:
            raise NotImplementedError
        if split == "celeba":
            dataset = CelebAMaskHQ(path, transform)
        else:
            dataset = ImageFolder(path, transform)
        return dataset

    def _get_ambiant(self, database, save_path):
        a = 1

    def _get_latent(self, database, save_path):

        self.vae = self.vae.to(self.device)
        data_loader = DataLoader(database, batch_size=128, shuffle=False)
        means, labels = [], []
        with torch.no_grad():
            for images, batch_labels in tqdm(data_loader, desc='Processing Dataset'):
                # for idx, (images, batch_labels) in enumerate(data_loader):
                if self.ae_name == 'stable_diff_14_aug_max_rot':
                    all_im = []
                    all_labels = []
                    for i in range(self.nb_aug):
                        all_im.append(tr_rot(images))
                        all_labels.append(batch_labels)
                    all_im = torch.stack(all_im)
                    batch_labels = torch.stack(all_labels).permute(dims=(1, 0))
                    all_im = torch.permute(all_im, dims=(1, 0, 2, 3, 4))
                    sz = all_im.size(0), all_im.size(1)
                    images = all_im.reshape(-1, 3, 128, 128)

                elif self.ae_name == 'stable_diff_14_aug_max_max':
                    all_im = []
                    all_labels = []
                    for i in range(self.nb_aug):
                        all_im.append(tr_max_2(images))
                        all_labels.append(batch_labels)
                    all_im = torch.stack(all_im)
                    batch_labels = torch.stack(all_labels).permute(dims=(1, 0))
                    all_im = torch.permute(all_im, dims=(1, 0, 2, 3, 4))
                    sz = all_im.size(0), all_im.size(1)
                    images = all_im.reshape(-1, 3, 128, 128)

                elif self.ae_name == 'stable_diff_14_aug_max':
                    all_im = []
                    all_labels = []
                    for i in range(self.nb_aug):
                        all_im.append(tr_max(images))
                        all_labels.append(batch_labels)
                    all_im = torch.stack(all_im)
                    batch_labels = torch.stack(all_labels).permute(dims=(1, 0))
                    all_im = torch.permute(all_im, dims=(1, 0, 2, 3, 4))
                    sz = all_im.size(0), all_im.size(1)
                    images = all_im.reshape(-1, 3, 128, 128)
                    # images = self.process.preprocess(all_im).to(self.device)
                    # outputs = self.vae.to(self.device).encode(images).latent_dist.mean
                    # outputs = outputs.view(sz[0], sz[1], 4, 16, 16)
                    a = 1
                elif self.ae_name == 'stable_diff_14_aug':
                    all_im = [images]
                    all_labels = [batch_labels]
                    images_flip = hflip(images)
                    all_im.append(images_flip)
                    all_labels.append(batch_labels)
                    for dis in range(1, 6):
                        im_rs = res_cro(images, top=dis * 3, left=dis * 3, width=128 - 2 * (dis * 3),
                                        height=128 - 2 * (dis * 3), size=(128, 128))
                        all_im.append(im_rs)
                        all_labels.append(batch_labels)
                        im_rs_flip = res_cro(images_flip, top=dis * 3, left=dis * 3, width=128 - 2 * (dis * 3),
                                             height=128 - 2 * (dis * 3), size=(128, 128))
                        all_im.append(im_rs_flip)
                        all_labels.append(batch_labels)
                    images = torch.cat(all_im)
                    batch_labels = torch.cat(all_labels)

                images = self.process.preprocess(images).to(self.device)
                outputs = self.vae.to(self.device).encode(images).latent_dist.mean
                if self.ae_name == 'stable_diff_14_aug_max' or self.ae_name == 'stable_diff_14_aug_max_max' or self.ae_name == 'stable_diff_14_aug_max_rot':
                    outputs = outputs.reshape(sz[0], sz[1], 4, 16, 16)
                means.append(outputs.detach().cpu())
                labels.append(batch_labels)
                # if idx == 2:
                #    break
        latent_tensor = torch.cat(means, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        mean = latent_tensor.mean()
        std = latent_tensor.std()
        latent_tensor = (latent_tensor - mean) / std
        latent_dict = {"latent": latent_tensor,
                       "label": label_tensor,
                       "mean": mean,
                       "std": std}

        torch.save(latent_dict, save_path)
        print(f"Latent data saved at: {save_path}")

        return latent_dict

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where t  arget is index of the target class.
        """
        if self.ae_name in ['stable_diff_14_aug_max', 'stable_diff_14_aug_max_max', 'stable_diff_14_aug_max_rot']:
            idx_aug = random.randint(0, self.nb_aug - 1)
            img = self.data_train["latent"][index, idx_aug]
            target = self.data_train["label"][index, idx_aug]
        else:
            img, target = self.data_train["latent"][index], self.data_train["label"][index]
        return img, target

    def __len__(self) -> int:
        return len(self.data_train["latent"])


class AlphaNum(VisionDataset):
    """`adatpted fom MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        data_root (string): Root directory of dataset where dataset.npz exists

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """

    def __init__(
            self,
            data_root: str,
            sequential: bool,
            one_letter: list = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ) -> None:
        super(AlphaNum, self).__init__(data_root, transform=transform,
                                       target_transform=target_transform)
        data = np.load(os.path.join(data_root, "dataset.npz"))
        # self.triplet = triplet
        self.seq = sequential
        if self.seq:
            self.angle_transition = torch.tensor([-10.0, 10.0])
        image = torch.tensor(data['data'])
        image = CenterCrop(size=(240, 240))(image)
        image = Resize(size=(32, 32))(image)
        label_name = list(data['label'])
        all_angle = torch.linspace(0, 358, 180)
        # all_angle = torch.linspace(0, 350, 36)
        if one_letter is not None:
            idx_letter = label_name.index(one_letter)
            image = image[idx_letter].unsqueeze(0)
            label_name = [label_name[idx_letter]]
        all_im = []

        all_label = []
        for each_angle in all_angle:
            all_im.append(rotate(image, float(each_angle), interpolation=Interp.BILINEAR))
            all_label += label_name

        self.image = torch.cat(all_im, dim=0).unsqueeze(1)
        self.label = all_label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.image[index], self.label[index]
        if self.seq:
            # orientation = self.angle_transition[random.randint(0,1)]
            orientation = self.angle_transition[0]
            img_t_plus = rotate(img, float(orientation))
            return img, img_t_plus, target
        else:
            return img, target

    def __len__(self) -> int:
        return len(self.image)


class AlphaNumV2(VisionDataset):
    """`adatpted fom MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        data_root (string): Root directory of dataset where dataset.npz exists

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """

    def __init__(
            self,
            data_root: str,
            sequential: bool,
            one_letter: list = None,
            rot_dist: str = 'uniform',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ) -> None:
        super(AlphaNumV2, self).__init__(data_root, transform=transform,
                                         target_transform=target_transform)
        data = np.load(os.path.join(data_root, "dataset.npz"))
        # self.triplet = triplet
        self.seq = sequential
        self.rot_dist = rot_dist
        if self.seq:
            self.angle_transition = torch.tensor([-10.0, 10.0])
        image = torch.tensor(data['data'])
        image = CenterCrop(size=(240, 240))(image)
        image = Resize(size=(32, 32))(image)
        label_name = list(data['label'])
        # all_angle = torch.linspace(0, 358, 180)
        all_angle = torch.linspace(-180, 178, 180)
        # all_angle = torch.linspace(0, 350, 36)

        all_im = []
        # all_label = []
        for each_angle in all_angle:
            all_im.append(rotate(image, float(each_angle), interpolation=Interp.BILINEAR))
            # all_label += label_name
        self.image = torch.stack(all_im, dim=1).unsqueeze(2)

        self.label = label_name

        if one_letter is not None:
            weight_dist_letter = torch.zeros(len(label_name))
            idx_letter = label_name.index(one_letter)
            weight_dist_letter[idx_letter] = 1.
        else:
            weight_dist_letter = torch.ones(len(label_name))
        self.dist_letter = dist.Categorical(weight_dist_letter)

        if self.rot_dist == 'gaussian':
            mean_angle = 0
            std_angle = 90
            weight_dist_angle = torch.exp(-0.5 * ((all_angle - mean_angle) / std_angle) ** 2)
        elif self.rot_dist == 'uniform':
            weight_dist_angle = torch.ones(len(all_angle))
        else:
            raise NotImplementedError()
        a = 1
        self.dist_rotation = dist.Categorical(weight_dist_angle)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        idx_letter = int(self.dist_letter.sample((1,)))
        idx_rot = int(self.dist_rotation.sample((1,)))

        img, target = self.image[idx_letter, idx_rot], self.label[idx_letter]
        if self.seq:
            # orientation = self.angle_transition[random.randint(0,1)]
            orientation = self.angle_transition[0]
            img_t_plus = rotate(img, float(orientation))
            return img, img_t_plus, target
        else:
            return img, target

    def __len__(self) -> int:
        return self.image.shape[0] * self.image.shape[1]


class CelebAMaskHQ():
    def __init__(self, img_path, transform_img, mode=True):
        self.img_path = img_path
        # self.label_path = label_path
        self.transform_img = transform_img
        # self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()

        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):

        for path in os.listdir(self.img_path):
            if os.path.isfile(os.path.join(self.img_path, path)) and path.endswith('.jpg') or path.endswith('.png'):
                img_path = os.path.join(self.img_path, path)
                # label_path = os.path.join(self.label_path, path.replace('.jpg', '.png'))
                # print(img_path, label_path)
                if self.mode == True:
                    # self.train_dataset.append([img_path, label_path])
                    self.train_dataset.append(img_path)
                # else:
                #    self.test_dataset.append([img_path, label_path])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):

        dataset = self.train_dataset if self.mode == True else self.test_dataset
        # img_path, label_path = dataset[index]
        img_path = dataset[index]
        image = Image.open(img_path)
        # label = Image.open(label_path)
        return self.transform_img(image), 0  # , self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images