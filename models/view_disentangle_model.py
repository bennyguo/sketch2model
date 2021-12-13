import os
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.loss import MSELoss
import torchvision
from tqdm import tqdm

import soft_renderer as sr
import soft_renderer.functional as srf


from .base_model import BaseModel
from .networks import init_net
from .criterions import *
from utils.utils import tensor2im

class gradient_reversal(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return gradient_reversal.apply(x, self.lambda_)


class ResNet18Encoder(nn.Module):
    def __init__(self, dim_in, pretrained):
        super(ResNet18Encoder, self).__init__()
        assert(dim_in == 3)
        print('ResNet18 pretrained:', pretrained)
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.x = {}

    def forward(self, x):
        batch_size = x.shape[0]
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0 = self.backbone.maxpool(x0)

        x1 = self.backbone.layer1(x0)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)  

        self.x[0], self.x[1], self.x[2], self.x[3], self.x[4] = x0, x1, x2, x3, x4

        x = x4
        x = x.view(batch_size, -1)
        return x


class Encoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_s, dim_v, normalize=True):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc_s = nn.Linear(dim_hidden, dim_s)
        self.fc_v = nn.Linear(dim_hidden, dim_v)
        self.normalize = normalize

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        zs = F.relu(self.fc_s(x), inplace=True)
        zv = F.relu(self.fc_v(x), inplace=True)
        if self.normalize:
            zs = F.normalize(zs, dim=1)
            zv = F.normalize(zv, dim=1)
        return zs, zv


class ViewEncoder(nn.Module):
    def __init__(self, dim_hidden, dim_out, normalize=True):
        super(ViewEncoder, self).__init__()
        dim_in = 2
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.normalize = normalize

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        if self.normalize:
            x = F.normalize(x, dim=1)
        return x


class ViewDecoder(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(ViewDecoder, self).__init__()
        dim_out = 2
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.sigmoid(self.fc2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, dim_in, dim_shape, dim_view, dim_hidden, normalize=True):
        super(Decoder, self).__init__()
        self.dim_in = dim_in
        self.dim_shape = dim_shape
        self.dim_view = dim_view
        self.fc1 = nn.Linear(dim_in + dim_view, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_shape)
        self.normalize = normalize

    def forward(self, x, view):
        x = torch.cat([x, view], dim=1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        if self.normalize:
            x = F.normalize(x, dim=1)
        return x



class MeshDecoder(nn.Module):
    """This MeshDecoder follows N3MR and SoftRas"""
    def __init__(self, filename_obj, dim_in, centroid_scale=0.1, bias_scale=1.0):
        super(MeshDecoder, self).__init__()
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, dim=self.dim)


class ViewDisentangleNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_extractor = ResNet18Encoder(dim_in=opt.dim_in, pretrained=True)
        self.encoder = Encoder(dim_in=512 * 7 * 7, dim_hidden=2048, dim_s=1024, dim_v=opt.view_dim, normalize=True)
        self.view_encoder = ViewEncoder(dim_hidden=opt.view_dim, dim_out=opt.view_dim, normalize=True)
        self.view_decoder = ViewDecoder(dim_in=opt.view_dim, dim_hidden=opt.view_dim//2)
        self.decoder = Decoder(dim_in=1024, dim_shape=512, dim_view=opt.view_dim, dim_hidden=1024, normalize=True)

        self.shape_discriminator = nn.Sequential(
            GradientReversalLayer(lambda_=opt.grl_lambda),
            nn.Linear(opt.n_vertices * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.domain_discriminator = nn.Sequential(
            GradientReversalLayer(lambda_=opt.grl_lambda),
            nn.Linear(25088, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.mesh_decoder = MeshDecoder(opt.template_path, dim_in=512)  

    def forward(self, image, view=None, view_rand=None):
        N = image.shape[0]
        if view is None and view_rand is None:
            """Inference"""
            ft = self.feature_extractor(image[:,:self.opt.dim_in])
            zs, zv_pred = self.encoder(ft)
            view_pred = self.view_decoder(zv_pred)
            zv_recon = self.view_encoder(view_pred)
            z = self.decoder(zs, zv_recon)
            vertices, faces = self.mesh_decoder(z)
            return {
                'vertices': vertices,
                'faces': faces,
                'view_pred': view_pred
            }
        else:
            """Training / Validation / Testing"""
            ft = self.feature_extractor(image[:,:self.opt.dim_in])
            zs, zv_pred = self.encoder(ft)
            view_pred = self.view_decoder(zv_pred)
            zv_recon = self.view_encoder(view_pred)
            zv = self.view_encoder(view)
            view_recon = self.view_decoder(zv)
            z = self.decoder(zs, zv) # teacher forcing
            z_pred = self.decoder(zs, zv_recon)
            vertices, faces = self.mesh_decoder(z)
            vertices_pred, faces_pred = self.mesh_decoder(z_pred)
            sd_score = self.shape_discriminator(vertices.view(N, -1))
            rv = {
                'vertices': vertices,
                'faces': faces,
                'vertices_pred': vertices_pred,
                'faces_pred': faces_pred,
                'view_pred': view_pred,
                'view_recon': view_recon,
                'zv_pred': zv_pred,
                'zv': zv,
                'zv_recon': zv_recon,
                'sd_score': sd_score
            }
            
            if view_rand is not None:
                """Training"""
                zv_rand = self.view_encoder(view_rand)
                z_rand = self.decoder(zs, zv_rand)
                vertices_rand, faces_rand = self.mesh_decoder(z_rand)
                sd_score_rand = self.shape_discriminator(vertices_rand.view(N, -1))
                rv.update({
                    'vertices_rand': vertices_rand,
                    'faces_rand': faces_rand,
                    'zv_rand': zv_rand,
                    'sd_score_rand': sd_score_rand
                })

            return rv


class ViewDisentangleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the tensorboard.
        self.train_loss_names = ['tot', 'iou_tot', 'iou_rand_tot', 'laplacian', 'flatten', 'view_pred', 'view_recon', 'zv_recon', 'sd']
        self.val_loss_names = ['voxel_iou', 'voxel_iou_pred']
        self.test_loss_names = ['voxel_iou', 'voxel_iou_pred']

        self.model_names = ['Full']
        self.netFull = init_net(ViewDisentangleNetwork(opt), opt)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            self.criterions = {
                'laplacian': LaplacianLoss(opt),
                'flatten': FlattenLoss(opt),
                'multiview-iou': MultiViewIoULoss(opt),
                'iou': IoULoss(opt),
                'mse': MSELoss(opt),
                'gan': F.binary_cross_entropy_with_logits
            }
            # define and initialize optimizers.
            self.optimizer = torch.optim.Adam(self.netFull.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # larger scales correspond to smaller rendered images
        # render_size = image_size // render_scale
        self.render_scales = [1, 2, 4]
        self.sigmas = [1e-5, 3e-5, 1e-4]
        self.adaptive_weighting_func = [
            lambda e: 1 if e > 1600 else 0,
            lambda e: 1 if 800 < e <= 1600 else 0,
            lambda e: 1 if e <= 800 else 0
        ]
        self.renderers = [
            sr.SoftRasterizer(
                image_size=opt.image_size // scale, sigma_val=sigma, aggr_func_rgb='hard', aggr_func_alpha='prod', dist_eps=1e-10
            ) for (scale, sigma) in zip(self.render_scales, self.sigmas)
        ]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        for k, v in input.items():
            if isinstance(v, list):
                """Training"""
                setattr(self, f"data_{k}_a", v[0].to(self.device))
                setattr(self, f"data_{k}_b", v[1].to(self.device))
            else:
                """Validation / Testing / Inference"""
                setattr(self, f"data_{k}", v.to(self.device))
    
    def update_hyperparameters(self, epoch):
        super().update_hyperparameters(epoch)
        self.adaptive_weighting = [f(epoch) for f in self.adaptive_weighting_func]

    def update_hyperparameters_step(self, step):
        super().update_hyperparameters_step(step)

    def get_random_view(self, N):
        """
        Get random elevation angle from [-20, 40]
        Get random azimuth angle from [-180, 180]
        """
        elevation_rand = (torch.rand(N, dtype=torch.float32) * 60 - 20)
        azimuth_rand = (torch.rand(N, dtype=torch.float32) * 360 - 180)
        elevation_rand, azimuth_rand = elevation_rand.to(self.device), azimuth_rand.to(self.device)
        return elevation_rand, azimuth_rand

    def encode_view(self, view):
        """
        Project elevation angle from [-20, 40] to [0, 1]
        Project azimuth angle from [-180, 180] to [0, 1]
        """
        view = view.clone()
        view[:,0] = (view[:,0] + 20) / 60.
        view[:,1] = (view[:,1] + 180) / 360.
        return view

    def decode_view(self, view):
        """
        Un-project elevation angle from [0, 1] to [-20, 40]
        Un-project azimuth angle from [0, 1] to [-180, 180]
        """        
        view = view.clone()
        view[:,0] = (view[:,0] * 60) - 20.
        view[:,1] = (view[:,1] * 360) - 180.
        return view

    def view2camera(self, view):
        """
        Caculate camera position from given elevation and azimuth angle.
        The camera looks at the center of the object, with a distance of 2.
        """
        N = view.shape[0]
        distance = torch.ones(N, dtype=torch.float32) * 2.
        distance = distance.to(self.device)
        camera = srf.get_points_from_angles(distance, view[:,0], view[:,1])
        return camera        
    
    def render_silhouette(self, v, f, camera, multiview=False):
        transform = sr.LookAt(viewing_angle=15, eye=camera)
        # only render when w > 0 to save time
        sil = [r(transform(sr.Mesh(v, f))) if w > 0 else None for r, w in zip(self.renderers, self.adaptive_weighting)]
        return [s.chunk(4, dim=0) if s is not None else None for s in sil] if multiview else sil

    def forward(self):
        """Run forward pass."""
        N = self.data_image_a.shape[0]

        image_ab = torch.cat([self.data_image_a, self.data_image_b], dim=0)
        self.data_image_ab = image_ab
        camera_aabb = torch.cat([self.data_camera_a, self.data_camera_a, self.data_camera_b, self.data_camera_b], dim=0)

        view_a = torch.cat([self.data_elevation_a[:, None], self.data_azimuth_a[:, None]], dim=1)
        view_b = torch.cat([self.data_elevation_b[:, None], self.data_azimuth_b[:, None]], dim=1)

        elevation_a_rand, azimuth_a_rand = self.get_random_view(N)
        elevation_b_rand, azimuth_b_rand = self.get_random_view(N)    

        view_a_rand = torch.cat([elevation_a_rand[:, None], azimuth_a_rand[:, None]], dim=1)
        view_b_rand = torch.cat([elevation_b_rand[:, None], azimuth_b_rand[:, None]], dim=1)
        camera_a_rand, camera_b_rand = self.view2camera(view_a_rand), self.view2camera(view_b_rand)
        camera_ab_rand = torch.cat([camera_a_rand, camera_b_rand], dim=0)

        view_a, view_b = self.encode_view(view_a), self.encode_view(view_b)
        view_a_rand, view_b_rand = self.encode_view(view_a_rand), self.encode_view(view_b_rand)

        view_ab = torch.cat([view_a, view_b], dim=0)
        view_ab_rand = torch.cat([view_a_rand, view_b_rand], dim=0)
        self.data_view_ab, self.data_view_ab_rand = view_ab, view_ab_rand

        out = self.netFull(image_ab, view=view_ab, view_rand=view_ab_rand)
        for k, v in out.items():
            setattr(self, f"out_{k}", v)
        
        self.out_silhouette = self.render_silhouette(
            torch.cat([self.out_vertices, self.out_vertices], dim=0),
            torch.cat([self.out_faces, self.out_faces], dim=0),
            camera_aabb,
            multiview=True
        )
        self.out_silhouette_rand = self.render_silhouette(
            self.out_vertices_rand,
            self.out_faces_rand,
            camera_ab_rand,
            multiview=False
        )
    
    def forward_test(self):
        """Run forward pass for validation / testing."""
        N = self.data_image.shape[0]
        self.data_view = torch.cat([self.data_elevation[:, None], self.data_azimuth[:, None]], dim=1)
        self.data_view = self.encode_view(self.data_view)
        out = self.netFull(self.data_image, view=self.data_view)
        for k, v in out.items():
            setattr(self, f"out_{k}", v)
        
    def forward_inference(self):
        """Run forward pass for inference."""
        if self.opt.view is None:
            self.data_view = None
        else:
            self.data_view = torch.cat([self.data_elevation[:, None], self.data_azimuth[:, None]], dim=1)
            self.data_view = self.encode_view(self.data_view)
        out = self.netFull(self.data_image, view=self.data_view)
        for k, v in out.items():
            setattr(self, f"out_{k}", v)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_laplacian, self.loss_laplacian_rand = self.criterions['laplacian'](self.out_vertices), self.criterions['laplacian'](self.out_vertices_rand)
        self.loss_flatten, self.loss_flatten_rand = self.criterions['flatten'](self.out_vertices), self.criterions['flatten'](self.out_vertices_rand)
        self.loss_view_pred = self.criterions['mse'](self.data_view_ab, self.out_view_pred)
        self.loss_view_recon = self.criterions['mse'](self.data_view_ab, self.out_view_recon)
        self.loss_zv_recon = self.criterions['mse'](self.out_zv_pred, self.out_zv_recon)
        self.loss_iou = [
            self.criterions['multiview-iou'](
                sil,
                F.interpolate(self.data_image_a, (self.opt.image_size//scale, self.opt.image_size//scale), mode='nearest'),
                F.interpolate(self.data_image_b, (self.opt.image_size//scale, self.opt.image_size//scale), mode='nearest')
            ) * w if w > 0 else 0 for sil, w, scale in zip(self.out_silhouette, self.adaptive_weighting, self.render_scales)
        ]
        self.loss_iou_tot = sum(self.loss_iou)
        self.loss_iou_rand = [
            self.criterions['iou'](
                sil,
                torch.cat([
                    F.interpolate(self.data_image_a, (self.opt.image_size//scale, self.opt.image_size//scale), mode='nearest'),
                    F.interpolate(self.data_image_b, (self.opt.image_size//scale, self.opt.image_size//scale), mode='nearest')
                ], dim=0)
            ) * w if w > 0 else 0 for sil, w, scale in zip(self.out_silhouette_rand, self.adaptive_weighting, self.render_scales)
        ]
        self.loss_iou_rand_tot = sum(self.loss_iou_rand)

        self.loss_sd_real = self.criterions['gan'](self.out_sd_score, torch.ones_like(self.out_sd_score))
        self.loss_sd_fake = self.criterions['gan'](self.out_sd_score_rand, torch.zeros_like(self.out_sd_score_rand))
        self.loss_sd = self.loss_sd_real + self.loss_sd_fake

        self.loss_tot = self.loss_iou_tot + self.opt.lambda_iou_rand * self.loss_iou_rand_tot + \
            self.opt.lambda_laplacian * (self.loss_laplacian + self.loss_laplacian_rand) + \
            self.opt.lambda_flatten * (self.loss_flatten + self.loss_flatten_rand) + \
            self.opt.lambda_view_pred * self.loss_view_pred + \
            self.opt.lambda_view_recon * self.loss_view_recon + \
            self.opt.lambda_zv_recon * self.loss_zv_recon + \
            self.opt.lambda_sd * self.loss_sd

        self.loss_tot.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
    
    def visualize_train(self, it):
        """Save generated meshes every opt.vis_freq training steps."""
        vis_save_dir = os.path.join(self.save_dir, 'train')
        os.makedirs(vis_save_dir, exist_ok=True)
        view = self.decode_view(self.data_view_ab)
        view_pred = self.decode_view(self.out_view_pred)
        view_rand = self.decode_view(self.data_view_ab_rand)
        image = self.data_image_ab[0]
        vt, f, vt_pred, f_pred, vt_rand, f_rand = self.out_vertices[0], self.out_faces[0], self.out_vertices_pred[0], self.out_faces_pred[0], self.out_vertices_rand[0], self.out_faces_rand[0]
        v, v_pred, v_rand = view[0], view_pred[0], view_rand[0]
        cv2.imwrite(os.path.join(vis_save_dir, f'{it:05d}_input.png'), tensor2im(image)[...,:3])
        srf.save_obj(os.path.join(vis_save_dir, f'{it:05d}_e{int(v[0])}a{int(v[1])}.obj'), vt, f)
        srf.save_obj(os.path.join(vis_save_dir, f'{it:05d}_pred_e{int(v_pred[0])}a{int(v_pred[1])}.obj'), vt_pred, f_pred)
        srf.save_obj(os.path.join(vis_save_dir, f'{it:05d}_rand_e{int(v_rand[0])}a{int(v_rand[1])}.obj'), vt_rand, f_rand)

    def validate(self, epoch, dataset, phase='val', save_dir=None):
        """Validation procedure. Called every opt.val_epoch_freq epochs."""
        count = 0
        iou_tot, iou_pred_tot = 0., 0.
        for i, data in enumerate(tqdm(dataset, desc=phase, total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward_test()
            voxel_gt = self.data_voxel.cpu().numpy()
            faces = srf.face_vertices(self.out_vertices, self.out_faces) * 31. / 32. + 0.5
            voxel = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[...,::-1]
            faces_pred = srf.face_vertices(self.out_vertices_pred, self.out_faces_pred) * 31. / 32. + 0.5
            voxel_pred = srf.voxelization(faces_pred, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[...,::-1]
            iou, iou_pred = voxel_iou(voxel, voxel_gt), voxel_iou(voxel_pred, voxel_gt)
            iou_tot += iou * self.data_image.shape[0]
            iou_pred_tot += iou_pred * self.data_image.shape[0]
            count += self.data_image.shape[0]

            if i < getattr(self.opt, f"{phase}_epoch_vis_n"):
                vis_save_dir = save_dir or os.path.join(self.save_dir, f"vis_{epoch}_{phase}")
                os.makedirs(vis_save_dir, exist_ok=True)
                view = self.decode_view(self.data_view)
                view_pred = self.decode_view(self.out_view_pred)
                image = self.data_image[0]
                vt, f, vt_pred, f_pred = self.out_vertices[0], self.out_faces[0], self.out_vertices_pred[0], self.out_faces_pred[0]
                v, v_pred = view[0], view_pred[0]
                cv2.imwrite(os.path.join(vis_save_dir, f'{i:02d}_input.png'), tensor2im(image)[...,:3])
                srf.save_obj(os.path.join(vis_save_dir, f'{i:02d}_e{int(v[0])}a{int(v[1])}.obj'), vt, f)
                srf.save_obj(os.path.join(vis_save_dir, f'{i:02d}_pred_e{int(v_pred[0])}a{int(v_pred[1])}.obj'), vt_pred, f_pred)

        self.loss_voxel_iou, self.loss_voxel_iou_pred = iou_tot / count, iou_pred_tot / count

    def test(self, epoch, dataset, save_dir=None):
        """Validation procedure. Called every opt.test_epoch_freq epochs."""
        self.validate(epoch, dataset, phase='test', save_dir=save_dir)

    def inference(self, epoch, dataset, save_dir):
        """Validation procedure. Generate 3D model from an input sketch and (optional) a given view."""
        data = next(iter(dataset))
        self.set_input(data)
        self.forward_inference()
        image = self.data_image[0]
        cv2.imwrite(os.path.join(save_dir, f'input.png'), tensor2im(image)[...,:3])
        if self.opt.view is None:
            v_pred = self.decode_view(self.out_view_pred)[0]
            vt, f = self.out_vertices[0], self.out_faces[0]
            srf.save_obj(os.path.join(save_dir, f'pred-view_e{int(v_pred[0])}a{int(v_pred[1])}.obj'), vt, f)
        else:
            v = self.decode_view(self.data_view)[0]
            v_pred = self.decode_view(self.out_view_pred)[0]
            vt, f, vt_pred, f_pred = self.out_vertices[0], self.out_faces[0], self.out_vertices_pred[0], self.out_faces_pred[0]
            srf.save_obj(os.path.join(save_dir, f'given-view_e{int(v[0])}a{int(v[1])}.obj'), vt, f)
            srf.save_obj(os.path.join(save_dir, f'pred-view_e{int(v_pred[0])}a{int(v_pred[1])}.obj'), vt_pred, f_pred)
