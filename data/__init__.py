"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
# or from .base_dataset import BaseDataset
# 虽然这里划红线，说在当前路径下找不到data文件夹，
# 但是我们在使用命令行运行的时候，是在整个项目的顶层运行命令
# 这时data文件就是同级文件，运行的时候就能识别到


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    #  dataset_name : default='unaligned'  - datamodel parameter
    datasetlib = importlib.import_module(dataset_filename)
    # then import unaligned_dataset.py in data

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items(): #
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            #  now dataset is  a  class  type : UnalignedDataset(BaseDataset)
            #  and basedata is inherited from data.dataset
            #  so actually  now dataset is a type of data.dataset ,
            #  so it can be input of dataloader

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data() # return self
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        #  opt.dataset_mode : default = 'unaligned'
        # e.g return a class  UnalignedDataset(BaseDataset)

        self.dataset = dataset_class(opt) # dataset_class ：UnalignedDataset
        # 返回一个继承于data.base_dataset 的 UnalignedDataset,
        # base_dataset又 继承于torch.utils.data.dataset的类
        # e.g in class UnalignedDataset，it __getitem__ function return
        # {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        print("dataset [%s] was created" % type(self.dataset).__name__)
        # 将得到的继承类传入到dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
