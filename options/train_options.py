from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.set_defaults(phase='train')

        parser.add_argument('--lambda_iou_rand', type=float, default=0.1)
        parser.add_argument('--lambda_laplacian', type=float, default=5e-3)
        parser.add_argument('--lambda_flatten', type=float, default=5e-4)
        parser.add_argument('--lambda_view_pred', type=float, default=10)
        parser.add_argument('--lambda_view_recon', type=float, default=10)
        parser.add_argument('--lambda_zv_recon', type=float, default=100)
        parser.add_argument('--lambda_sd', type=float, default=0.1)

        # visdom and HTML visualization parameters
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--vis_freq', type=int, default=100, help='training visualization frequency, in steps')
        parser.add_argument('--val_epoch_freq', type=int, default=50, help='validation frequency, in epochs')
        parser.add_argument('--val_epoch_vis_n', type=int, default=20, help='number of data to visualize in validation')
        parser.add_argument('--test_epoch_freq', type=int, default=50, help='testing frequency, in epochs')
        parser.add_argument('--test_epoch_vis_n', type=int, default=20, help='number of data to visualize in testing')

        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load model at epoch [load_epoch]')
        parser.add_argument('--init_weights', type=str, default=None, help='initialize weights from an existing model, in format [name]:[epoch]')
        parser.add_argument('--init_weights_keys', type=str, default='.+', help='regex for weights keys to be loaded')
        parser.add_argument('--fix_layers', type=str, default=None, help='regex for fix layers')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs in total')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | exp | step]')

        # linear | exp policy
        parser.add_argument('--lr_final', type=float, default=1e-5, help='final learning rate for adam, used in linear and exp')
        parser.add_argument('--n_epochs_decay', type=int, default=1000, help='number of epochs to decay learning rate to lr_final')

        # step policy
        parser.add_argument('--lr_decay_epochs', type=int, default=800, help='multiply by a gamma every lr_decay_epochs epochs, used in step')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.3, help='multiply by a gamma every lr_decay_epochs epochs, used in step')

        self.isTrain, self.isTest, self.isInfer = True, False, False
        return parser
