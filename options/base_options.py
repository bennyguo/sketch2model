import argparse
from models.base_model import BaseModel
from data.base_dataset import BaseDataset
import os
from utils import utils
import torch
import options
import json
from utils.utils import find_class_using_name


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--summary_dir', type=str, default='./runs', help='tensorboard logs are saved here')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--class_id', type=str, required=True)

        # model parameters
        parser.add_argument('--model', type=str, default='view_disentangle', choices=['view_disentangle'], help='chooses which model to use.')
        parser.add_argument('--dim_in', type=int, default=3, help='number of input channels for image feature extractor')
        parser.add_argument('--grl_lambda', type=float, default=1, help='lambda in gradient reversal layer')
        parser.add_argument('--n_vertices', type=int, default=642, help='number of vertices of the base mesh')
        parser.add_argument('--image_size', type=int, default=224, help='input image size')
        parser.add_argument('--view_dim', type=int, default=512, help='dimension of the view latent code')
        parser.add_argument('--template_path', type=str, default='templates/sphere_642.obj', help='path to the base mesh')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='shapenet', choices=['shapenet', 'shapenet_sketch', 'inference'], help='chooses how datasets are loaded.')
        parser.add_argument('--dataset_root', type=str, default='load/shapenet-synthetic')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # additional parameters
        parser.add_argument('--phase', type=str, choices=['train', 'test', 'infer'])
        parser.add_argument('--load_epoch', type=str, default='latest', help='epoch to load')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{batch_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = options.get_option_setter(find_class_using_name(f"models.{model_name}_model", model_name, 'model', BaseModel))
        parser = model_option_setter(parser)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = options.get_option_setter(find_class_using_name(f"data.{dataset_name}_dataset", dataset_name, 'dataset', BaseDataset))
        parser = dataset_option_setter(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file -- [checkpoints_dir] / opt.txt
        It will save options into a json file -- [checkpoints_dir] / opt.json
        """
        message = ''
        message += '----------------- Options ---------------\n'
        opt_dict = {}
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            opt_dict[k] = v
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        with open(os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase)), 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        with open(os.path.join(expr_dir, '{}_opt.json'.format(opt.phase)), 'w') as opt_json_file:
            opt_json_file.write(json.dumps(opt_dict))

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.isTest = self.isTest
        opt.isInfer = self.isInfer

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        opt.n_gpus = torch.cuda.device_count()
        
        opt.device = 'cuda:0' if opt.n_gpus > 0 else 'cpu'
        
        if opt.n_gpus > 0:
            torch.cuda.set_device(opt.device)

        self.print_options(opt)

        self.opt = opt
        return self.opt
