import argparse
import os
import torch

class Config():
    """configions class

    Returns:
        [argparse]: argparse containing train and test configions
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')

        self.parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        # self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. plane  plane,1,2, plane,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        # self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='ganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost",
                                 help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        # self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--display', type=str, default=True, help='Use visdom.')

        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')

        self.parser.add_argument('--normaly_class', default='plane', help='Anomaly class idx for mnist and cifar datasets,mnist plane-9,cifar100 plane-20')
        self.parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of anomalies in test set.')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        # data
        self.parser.add_argument('--dataset', default='cifar10', help=' cifar10 | mnist | cifar100 ')
        self.parser.add_argument('--dataroot', default='./train1', help='path to dataset')
        self.parser.add_argument('--image_size', type=int, default=32)
        self.parser.add_argument('--patch_size', type=int, default=8)
        # channels and nc is input image channels
        self.parser.add_argument('--channels', type=int, default=3)
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')

        # device
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')

        self.parser.add_argument('--patch_dim', type=int, default=64)
        self.parser.add_argument('--pixel_dim', type=int, default=10)
        self.parser.add_argument('--pixel_size', type=int, default=4)
        self.parser.add_argument('--depth', type=int, default=6)
        self.parser.add_argument('--num_classes', type=int, default=1)
        self.parser.add_argument('--heads', type=int, default=8)
        self.parser.add_argument('--dim_head', type=int, default=256)
        self.parser.add_argument('--ff_dropout', type=float, default=0.5)
        self.parser.add_argument('--attn_dropout', type=float, default=0.5)
        self.parser.add_argument('--unfold_args', type=str, default=None)
        self.parser.add_argument('--n_extra_layers', type=int, default=6)





        # Train
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100,
                                 help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr_g', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_d', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument('--w_adv', type=float, default=1, help='Adversarial loss weight')
        self.parser.add_argument('--w_con', type=float, default=1, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=50, help='Encoder loss weight.')
        self.isTrain = True
        self.config = None


    def parse(self):
        """ Parse Arguments.
        """

        # self.config = self.parser.parse_args()
        self.config = self.parser.parse_known_args()[0]
        self.config.isTrain = self.isTrain   # train or test

        str_ids = self.config.gpu_ids.split(',')
        self.config.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.config.gpu_ids.append(id)

        # set gpu ids
        if self.config.device == 'gpu':
            torch.cuda.set_device(self.config.gpu_ids[0])

        args = vars(self.config)

        # print('------------ configions -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        if self.config.name == 'experiment_name':
            self.config.name = "%s/%s" % (self.config.model, self.config.dataset)
        expr_dir = os.path.join(self.config.outf, self.config.name, 'train')
        test_dir = os.path.join(self.config.outf, self.config.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'config.txt')
        with open(file_name, 'wt') as config_file:
            config_file.write('------------ configions -------------\n')
            for k, v in sorted(args.items()):
                config_file.write('%s: %s\n' % (str(k), str(v)))
            config_file.write('-------------- End ----------------\n')
        return self.config
