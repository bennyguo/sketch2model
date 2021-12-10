import os
import random
import torch
import torchvision.transforms.functional as TF
import numpy as np
from scipy.io import loadmat
from PIL import Image
from data.base_dataset import BaseDataset
import soft_renderer as sr
import soft_renderer.functional as srf


class ShapeNetSketchDataset(BaseDataset):
    """
    Dataset for loading ShapeNet-Sketch data.
    ShapeNet-Sketch is for evaluation only.
    """
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, mode):
        super().__init__(opt, mode)
        self.opt = opt
        self.root = os.path.join(opt.dataset_root, opt.class_id)
        self.class_id = opt.class_id

        self.dat = []
        for obj_id in os.listdir(self.root):
            obj_path = os.path.join(self.root, obj_id)
            with open(os.path.join(obj_path, 'view.txt')) as f:
                obj_cameras = [list(map(float, c.split(' '))) for c in list(filter(None, f.read().split('\n')))]
            obj_camera = obj_cameras[0]
            elevation, azimuth = self.get_view_tensor(obj_camera[1], obj_camera[0])
            self.dat.append({
                'class_id': self.class_id,
                'obj_id': obj_id,
                'image': os.path.join(obj_path, 'sketch.png'),
                'camera': self.get_camera_tensor(obj_camera[3], obj_camera[1], obj_camera[0]), # distance, elevation, azimuth
                'elevation': elevation,
                'azimuth': azimuth,
                'voxel': os.path.join(obj_path, 'voxel.mat'),
            })
        
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, index):
        view_dat = self.dat[index]
        camera = view_dat['camera']
        image = self.get_image_tensor(view_dat['image'])
        elevation, azimuth = view_dat['elevation'], view_dat['azimuth']
        voxel = self.get_voxel_tensor(view_dat['voxel'])
        return {
            'image': image,
            'camera': camera,
            'elevation': elevation,
            'azimuth': azimuth,
            'voxel': voxel
        }

    def get_image_tensor(self, path):
        image = Image.open(path).convert('RGBA')
        image = TF.resize(image, (self.opt.image_size, self.opt.image_size))
        image = TF.to_tensor(image)
        torch.where(image[3] > 0.5, torch.tensor(1.), torch.tensor(0.))
        return image
    
    def get_camera_tensor(self, distance, elevation, azimuth):
        camera = srf.get_points_from_angles(distance, elevation, azimuth)
        return torch.Tensor(camera).float()

    def get_voxel_tensor(self, path):
        # ground truth voxel head to x, up to y
        # transform to be able to compare with the voxel converted by SoftRas
        voxel = loadmat(path)['Volume'].astype(np.float32)
        voxel = np.rot90(np.rot90(voxel, axes=(1, 0)), axes=(2, 1))
        voxel = torch.from_numpy(voxel)
        return voxel

    def get_view_tensor(self, elevation, azimuth):
        return torch.tensor(elevation, dtype=torch.float32), torch.tensor(azimuth, dtype=torch.float32)
