from torchvision import datasets, transforms
import torch
import os
import glob
import pandas as pd
from PIL import Image


class IMetaDL_Dataset(torch.utils.data.Dataset):
    """
    Under root, there should be 
        labels.csv
        images/
    """
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform

        labels_path = os.path.join(root, "labels.csv")
        assert os.path.exists(labels_path)
        df = pd.read_csv(labels_path)
        df = df.rename({"newfilename": "FILE_NAME", "category": "CATEGORY"}, axis=1)
        df = df.rename({"uniquefilename": "FILE_NAME", "category": "CATEGORY"}, axis=1)

        df = df.loc[:, ["FILE_NAME", "CATEGORY"]]

        self.items = []
        raw_labels = dict()
        label_cnt = 0
        for index, row in df.iterrows():
            img_path = os.path.join(root, "images", row["FILE_NAME"])
            raw_label = row["CATEGORY"]
            self.items.append([img_path, raw_label])
            if raw_label not in raw_labels:
                raw_labels[raw_label] = label_cnt
                label_cnt += 1
        for item in self.items:
            item.append(raw_labels[item[1]])

        # this is not well documented, but this property is present in ImageFolder
        self.classes = sorted(list(raw_labels.keys()))


    def __getitem__(self, index):
        img_path = self.items[index][0]
        img = Image.open(img_path).convert("RGB")

        target = self.items[index][2]

        if self.transform is not None:
            img = self.transform(img) 
        
        return img, target

    def __len__(self):
        return len(self.items)

def load_data(data_folder, batch_size, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    data = IMetaDL_Dataset(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class


def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0







