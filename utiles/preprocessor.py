from fastai.data.external import untar_data, URLs
import os
import glob
import numpy as np
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tqdm
import cv2

SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


def sample_test_dataset(coco_path, images_count=20000):
    """
    The function that downloads the basic dataset from COCO. The dataset includes up to 20,000 images.
    :param coco_path: Path local to the coco dataset
    :param images_count: The number of images in the dataset. Maximum: 20,000. Minimum: 3,000
    :return:
    """
    if not os.path.exists(coco_path):
        coco_path = untar_data(URLs.COCO_SAMPLE)

    path = str(coco_path) + "/train_sample"
    paths = glob.glob(path + "/*.jpg")

    np.random.seed(224)
    paths_subset = np.random.choice(paths, images_count, replace=False)
    rand_idxs = np.random.permutation(images_count)
    train_idxs = rand_idxs[:images_count - 2000]
    val_idxs = rand_idxs[images_count - 2000:]
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    print(len(train_paths), len(val_paths))
    return train_paths, val_paths


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):
    """
    Dataloader function to load data and process it
    :param batch_size: Number of batch. Default: 16
    :param n_workers: Number of working thread for parallel job. Default: 4
    :param pin_memory: If pin memory. Default: True
    :param kwargs:
    :return:
    """
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader


def create_grayscale_image(path):
    path = str(path) + "/train_sample"
    paths = glob.glob(path + "/*.jpg")

    img = cv2.imread(paths[5])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('DR.png', img_gray)


def create_grayscale_my_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('DR.png', img_gray)


if __name__ == "__main__":
    # train_paths, val_paths = sample_test_dataset("/home/oleh/.fastai/data/coco_sample", 3000)
    # train_dl = make_dataloaders(paths=train_paths, split='train')
    # val_dl = make_dataloaders(paths=val_paths, split='val')
    #
    # data = next(iter(train_dl))
    # Ls, abs_ = data['L'], data['ab']
    # print(Ls.shape, abs_.shape)
    # print(len(train_dl), len(val_dl))

    # create_grayscale_image("/home/mshuiak/.fastai/data/coco_sample")
    create_grayscale_my_image("/home/mshuiak/Downloads/do_operu.jpg")
