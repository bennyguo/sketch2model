import torch
import torchvision.transforms.functional as TF
from PIL import Image
from data.base_dataset import BaseDataset


class InferenceDataset(BaseDataset):
    """
    Dataset for inferencing, containing only a single image with (optional) given view.
    """
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, mode):
        super().__init__(opt, mode)
        self.opt = opt
        self.image = self.get_image_tensor(opt.image_path)
        if opt.view is None:
            self.elevation, self.azimuth = 0, 0
        else:
            self.elevation, self.azimuth = torch.tensor(opt.view[0], dtype=torch.float32), torch.tensor(opt.view[1], dtype=torch.float32)
            assert -20 <= self.elevation <= 40 and -180 <= self.azimuth <= 180

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return {
            'image': self.image,
            'elevation': self.elevation,
            'azimuth': self.azimuth,
        }

    def get_image_tensor(self, path):
        image = Image.open(path).convert('RGBA')
        image = TF.resize(image, (self.opt.image_size, self.opt.image_size))
        image = TF.to_tensor(image)
        return image
