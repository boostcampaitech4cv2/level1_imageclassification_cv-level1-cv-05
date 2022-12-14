import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
import random

import numpy as np
from sympy import Not
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
from torchvision.transforms.functional import crop

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            # CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            # AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 58:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    def __init__(self, data_dir, rembg_dir,  usebbox, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.total_labels = []
        self.bb_paths = []
        self.train_idxs_in_dataset = None
        self.val_idxs_in_dataset = None

        self.data_dir = data_dir
        self.rembg_dir = rembg_dir
        self.usebbox = usebbox
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.bb_dir = data_dir.replace("images", "boundingbox")

        self.classes_hist = np.zeros(self.num_classes)
        self.transform = None
        if (os.path.isdir(os.path.join(self.rembg_dir))):
            self.userembg = True
        else:
            self.userembg = False

        self.setup()
        self.calc_statistics()

        if self.usebbox == 'yes':
            bb_path = os.path.join(self.bb_dir)
            print("use bounding box with this path : ", bb_path)

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            if self.userembg:
                img_folders = [os.path.join(self.data_dir, profile), os.path.join(
                    self.rembg_dir, profile)]
            else:
                img_folders = [os.path.join(self.data_dir, profile)]

            for idx_folder, img_folder in enumerate(img_folders):
                if idx_folder == 0:
                    folder_dir = self.data_dir
                else:
                    folder_dir = self.rembg_dir

                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    img_path = os.path.join(folder_dir, profile, file_name)

                    mask_label = self._file_names[_file_name]

                    if self.usebbox == 'yes':
                        # (resized_data, 000004_male_Asian_54, mask1.txt)
                        bb_path = os.path.join(
                            self.bb_dir, profile, _file_name + ".txt")

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    total_label = self.encode_multi_class(
                        mask_label, gender_label, age_label)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.total_labels.append(total_label)

                    if self.usebbox == 'yes':
                        self.bb_paths.append(bb_path)

                    self.classes_hist[total_label] = self.classes_hist[total_label] + 1

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(
            mask_label, gender_label, age_label)
        if self.usebbox == 'yes':
            bbox = self.read_boundingbox(index)
            if bbox is None:  # default : center crop
                bbox = [0, 0, 256, 320]
                bbox[0] = (384 - bbox[2])//2  # x
                bbox[1] = (512 - bbox[3])//2  # y

            image_transform = self.transform(
                crop(image, bbox[1], bbox[0], bbox[3], bbox[2]))
        else:
            image_transform = self.transform(image)

        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    def read_boundingbox(self, index):
        bb_path = self.bb_paths[index]
        bbox = None
        if (os.path.isfile(bb_path)):
            bboxfile = open(bb_path, 'r')
            bboxcoord = bboxfile.read().split(',', maxsplit=4)
            bbox = []
            for i in range(4):
                bbox.append(int(bboxcoord[i]))
            bboxfile.close()

        return bbox

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)

        indices_rand = torch.randperm(len(self))

        val_set_indices = indices_rand[:n_val]
        train_set_indices = indices_rand[n_val:]

        train_set = Subset(self, train_set_indices)
        val_set = Subset(self, val_set_indices)

        self.train_idxs_in_dataset = train_set_indices
        self.val_idxs_in_dataset = val_set_indices

        return train_set, val_set


class MaskStratifiedDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 class의 비율을 유지하면서 나눕니다.
    """

    def __init__(self, data_dir, rembg_dir, usebbox, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, rembg_dir, usebbox, mean, std, val_ratio)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        indices_per_label = defaultdict(list)
        for index, label in enumerate(self.total_labels):
            indices_per_label[label].append(index)
        val_set_indices, train_set_indices = list(), list()
        for label, indices in indices_per_label.items():
            n_samples_for_label = round(len(indices) * self.val_ratio)
            random_indices_sample = random.sample(indices, n_samples_for_label)
            val_set_indices.extend(random_indices_sample)
            train_set_indices.extend(set(indices) - set(random_indices_sample))
        train_set = Subset(self, train_set_indices)
        val_set = Subset(self, val_set_indices)
        # first_set_labels = list(map(self.total_labels.__getitem__, first_set_indices))
        self.train_idxs_in_dataset = train_set_indices
        self.val_idxs_in_dataset = val_set_indices
        # secound_set_labels = list(map(self.total_labels.__getitem__, second_set_indices))
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, rembg_dir, usebbox, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, rembg_dir, usebbox, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [
            profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                if self.userembg:
                    img_folders = [os.path.join(self.data_dir, profile), os.path.join(
                        self.rembg_dir, profile)]
                else:
                    img_folders = [os.path.join(self.data_dir, profile)]

                for idx_folder, img_folder in enumerate(img_folders):
                    if idx_folder == 0:
                        folder_dir = self.data_dir
                    else:
                        folder_dir = self.rembg_dir
                    for file_name in os.listdir(img_folder):
                        _file_name, ext = os.path.splitext(file_name)
                        if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                            continue

                        # (resized_data, 000004_male_Asian_54, mask1.jpg)
                        img_path = os.path.join(folder_dir, profile, file_name)
                        mask_label = self._file_names[_file_name]

                        if self.usebbox == 'yes':
                            # (resized_data, 000004_male_Asian_54, mask1.txt)
                            bb_path = os.path.join(
                                self.bb_dir, profile, _file_name + ".txt")

                        id, gender, race, age = profile.split("_")
                        gender_label = GenderLabels.from_str(gender)
                        age_label = AgeLabels.from_number(age)

                        total_label = self.encode_multi_class(
                            mask_label, gender_label, age_label)

                        self.image_paths.append(img_path)
                        self.mask_labels.append(mask_label)
                        self.gender_labels.append(gender_label)
                        self.age_labels.append(age_label)
                        self.total_labels.append(total_label)

                        if self.usebbox == 'yes':
                            self.bb_paths.append(bb_path)

                        self.classes_hist[total_label] = self.classes_hist[total_label] + 1

                        self.indices[phase].append(cnt)
                        cnt += 1

                        if self.usebbox == 'yes':
                            self.bb_paths.append(bb_path)

    def split_dataset(self) -> List[Subset]:
        self.train_idxs_in_dataset = self.indices["train"]
        self.val_idxs_in_dataset = self.indices["val"]

        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, bb_paths, resize, usebbox, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            # CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.usebbox = usebbox
        if self.usebbox == 'yes':
            self.bb_paths = bb_paths

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.usebbox == 'yes':
            bbox = self.read_boundingbox(index)
            if bbox is None:  # default : center crop
                bbox = [0, 0, 256, 320]
                bbox[0] = (384 - bbox[2])//2  # x
                bbox[1] = (512 - bbox[3])//2  # y
            image = crop(image, bbox[1], bbox[0], bbox[3], bbox[2])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

    def read_boundingbox(self, index):
        bb_path = self.bb_paths[index].replace("jpg", "txt")
        bbox = None
        if (os.path.isfile(bb_path)):
            bboxfile = open(bb_path, 'r')
            bboxcoord = bboxfile.read().split(',', maxsplit=4)
            bbox = []
            for i in range(4):
                bbox.append(int(bboxcoord[i]))
            bboxfile.close()
        return bbox
