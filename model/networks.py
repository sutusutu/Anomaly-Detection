

import torch.nn as nn
import torch
import tool.tnt as tnt


class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.conv_base_channel = 64

        self.tnt1 = tnt.TNT(image_size=config.image_size,
                            patch_dim=config.patch_dim,
                            pixel_dim=config.pixel_dim,
                            patch_size=config.patch_size,
                            pixel_size=config.pixel_size,
                            depth=config.depth,
                            num_classes=config.num_classes,
                            heads=config.heads,
                            dim_head=config.dim_head,
                            ff_dropout=config.ff_dropout,
                            attn_dropout=config.attn_dropout,
                            unfold_args=config.unfold_args,
                            channels=config.channels)

        self.linear_base_dim_1 = (config.image_size // config.patch_size) ** 2 + 1
        self.linear_base_dim_2 = (config.image_size // config.patch_size)
        self.to_encoder = nn.Sequential(
            nn.Linear(self.linear_base_dim_1 * config.patch_dim, self.linear_base_dim_2 * config.patch_dim),
            nn.LayerNorm(self.linear_base_dim_2 * config.patch_dim),
            nn.LeakyReLU(),
            nn.Linear(self.linear_base_dim_2 * config.patch_dim, self.linear_base_dim_2 // 2 * config.patch_dim),
            nn.Tanh(),
            nn.Linear(self.linear_base_dim_2 // 2 * config.patch_dim, config.patch_dim),
            nn.Tanh()
            )

        assert config.patch_dim == config.nz
        self.de = Decoder(isize=config.isize, nz=config.nz, nc=config.nc, ngf=config.ngf, ngpu=config.ngpu,
                          n_extra_layers=config.n_extra_layers)

        '''encoder2'''
        self.tnt2 = tnt.TNT(image_size=config.image_size,
                            patch_dim=config.patch_dim,
                            pixel_dim=config.pixel_dim,
                            patch_size=config.patch_size,
                            pixel_size=config.pixel_size,
                            depth=config.depth,
                            num_classes=config.num_classes,
                            heads=config.heads,
                            dim_head=config.dim_head,
                            ff_dropout=config.ff_dropout,
                            attn_dropout=config.attn_dropout,
                            unfold_args=config.unfold_args,
                            channels=config.channels)

        self.en2_trans = nn.Sequential(
            nn.Linear(self.linear_base_dim_1 * config.patch_dim, self.linear_base_dim_2 * config.patch_dim),
            nn.LayerNorm(self.linear_base_dim_2 * config.patch_dim),
            nn.LeakyReLU(),
            nn.Linear(self.linear_base_dim_2 * config.patch_dim, self.linear_base_dim_2 // 2 * config.patch_dim),
            nn.Tanh(),
            nn.Linear(self.linear_base_dim_2 // 2 * config.patch_dim, config.patch_dim),
            nn.Tanh()
            )

    def forward(self, input):
        en1 = self.tnt1(input)[1]
        to_decoder = en1.view(input.size()[0], (self.linear_base_dim_1 * self.config.patch_dim))
        de1 = self.to_encoder(to_decoder)
        latent1 = de1.view(input.size()[0], de1.size()[1], 1, 1)
        de = self.de(latent1)
        en2 = self.tnt2(de)[1]
        en2 = en2.view(input.size()[0], (self.linear_base_dim_1 * self.config.patch_dim))
        latent2 = self.en2_trans(en2).view(input.size()[0], de1.size()[1], 1, 1)

        return de, latent1, latent2


class Discriminate(nn.Module):
    def __init__(self, config):
        super(Discriminate, self).__init__()
        self.config = config

        self.tnt1 = tnt.TNT(image_size=config.image_size,
                            patch_dim=config.patch_dim,
                            pixel_dim=config.pixel_dim,
                            patch_size=config.patch_size,
                            pixel_size=config.pixel_size,
                            depth=config.depth,
                            num_classes=config.num_classes,
                            heads=config.heads,
                            dim_head=config.dim_head,
                            ff_dropout=config.ff_dropout,
                            attn_dropout=config.attn_dropout,
                            unfold_args=config.unfold_args,
                            channels=config.channels)


        self.linear_base_dim_1 = (config.image_size // config.patch_size) ** 2 + 1
        self.linear_base_dim_2 = (config.image_size // config.patch_size)
        self.en1_trans = nn.Sequential(
            nn.Linear(self.linear_base_dim_1 * config.patch_dim, self.linear_base_dim_2 * config.patch_dim),
            nn.LayerNorm(self.linear_base_dim_2 * config.patch_dim),
            nn.LeakyReLU(),
            nn.Linear(self.linear_base_dim_2 * config.patch_dim, self.linear_base_dim_2 // 2 * config.patch_dim),
            nn.Tanh(),
            nn.Linear(self.linear_base_dim_2 // 2 * config.patch_dim, config.patch_dim),
            nn.Tanh()
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        classifier, feature = self.tnt1(input)
        classifier = self.sigmoid(classifier).view(input.size()[0])
        feature = feature.view(input.size()[0], (self.linear_base_dim_1 * self.config.patch_dim))
        feature = self.en1_trans(feature)
        feature = feature.view(input.size()[0], feature.size()[1], 1, 1)
        return classifier, feature
