import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.base_folder = base_folder
        self.train = train
        self.p = p
        self.transforms = transforms

        if self.train:
            self.data_files = [os.path.join(base_folder, f'data_batch_{i}') for i in range(1, 6)]
        else:
            self.data_files = [os.path.join(base_folder, 'test_batch')]

        X = []
        y = []
        for file in self.data_files:
            data_dict = self._load_data(file)
            # 数据批次文件是以二进制格式存储的，字典的键值对如下：
            # 其中b'data': 包含了图像数据，b'labels': 包含了对应的标签
            X.append(data_dict[b'data'])
            y.append(data_dict[b'labels'])
        
        # np.concatenate(X, axis=0)：将列表 X 中的所有 numpy 数组按行连接成一个单一的数组
        self.X = np.concatenate(X, axis=0).reshape(-1, 3, 32, 32) / 255.  # CIFAR-10的图是32x32
        self.y = np.concatenate(y, axis=None)

        ### END YOUR SOLUTION

    def _load_data(self, file):  # 传入文件夹
        # pickle.load() 函数会将文件中的数据读取出来，
        # 并根据指定的 encoding='bytes' 参数将字节对象转换为 Python 对象
        import pickle
        with open(file, 'rb') as f:
            data_dict = pickle.load(f, encoding = 'bytes')
        return data_dict

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.transforms:
            X_items = np.array([self.apply_transforms(x) for x in self.X[index]])
        else:
            X_items = self.X[index]
        y_items = self.y[index]

        return X_items, y_items
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return len(self.y)
        ### END YOUR SOLUTION

