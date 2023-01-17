from collections import OrderedDict
import math
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from data.biomassters_dataset import BioMasstersDataset
from util import biomassters_utils as butils


class Pix2PixBioModel(BaseModel):
    """ This class implements the pix2pix model, adapted for the BioMassters dataset, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=0, help='weight for L2 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_L2', 'D_real', 'D_fake', 'RMSE', 'acc_D_fake', 'acc_D_real']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.g_activation)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        self.Y_SCALE = BioMasstersDataset(opt).Y_SCALE

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_L2
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        visual_ret['fake_B_L1_err'] = abs(getattr(self, 'fake_B') - getattr(self, 'real_B')) * self.Y_SCALE
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
                # DEBUG
                # butils.summarize_data(visual_ret[name], f'Raw visuals: {name}')
            if name == 'real_B':
                visual_ret[name+'_clip'] = butils.rescale_image(visual_ret[name], input_domain=[0,1], output_domain=[-1, 1], clip_input=True)
                visual_ret[name] = butils.rescale_image(visual_ret[name], input_domain=[0,1], output_domain=[-1, 1])
            elif name == 'fake_B':
                visual_ret[name] = torch.clamp(visual_ret[name], min=0, max=1)
                visual_ret[name+'_clip'] = butils.rescale_image(visual_ret[name], input_domain=[0,1], output_domain=[-1, 1], clip_input=True)
                visual_ret[name] = butils.rescale_image(visual_ret[name], input_domain=[0,1], output_domain=[-1, 1])
            elif name == 'real_A':
                visual_ret[name] = butils.rescale_image(visual_ret[name], input_domain=None, output_domain=[-1, 1])
            # DEBUG
            # butils.summarize_data(visual_ret[name], f'Inverse transformed visuals: {name}')
        return visual_ret
    
    def calculate_RMSE(self):
        """Calculate RMSE of generated image against ground truth"""
        fake_B = butils.rescale_image(self.fake_B, input_domain=[-1, 1], output_domain=[0, self.Y_SCALE])
        real_B = butils.rescale_image(self.real_B, input_domain=[-1, 1], output_domain=[0, self.Y_SCALE])

        # DEBUG
        # butils.summarize_data(fake_B, 'RMSE fake image')
        # butils.summarize_data(real_B, 'RMSE real image')

        criterionMSE = torch.nn.MSELoss()
        self.loss_RMSE = torch.sqrt(criterionMSE(fake_B, real_B)).cpu().detach().numpy()

    def return_distributions(self, log_transform=False):
        """
        Return generated image pixel distributions

        Parameters:
            log_transform (boolean): apply log transform

        Returns:
            tuple of histograms (tuple of np.ndarray)
        """
        # DEBUG
        # butils.summarize_data(self.fake_B, f'Dist raw: fake')
        # butils.summarize_data(self.real_B, f'Dist raw: real')
        fake_B = butils.rescale_image(torch.clamp(self.fake_B, min=0, max=1).detach().cpu().numpy(), input_domain=[0, 1], output_domain=[0, self.Y_SCALE])
        # fake_B = butils.rescale_image(self.fake_B.detach().cpu().numpy(), input_domain=[0, 1], output_domain=[0, self.Y_SCALE])
        real_B = butils.rescale_image(self.real_B.detach().cpu().numpy(), input_domain=[0, 1], output_domain=[0, self.Y_SCALE])
        # butils.summarize_data(self.fake_B, f'Dist inverted: fake')
        # butils.summarize_data(self.real_B, f'Dist inverted: real')

        hist_scale = self.Y_SCALE
        if log_transform:
            fake_B = np.log(1 + fake_B)
            real_B = np.log(1 + real_B)
            hist_scale = np.log(1+self.Y_SCALE*2)    
        fake_B_hist = np.histogram(fake_B, bins=np.arange(0, hist_scale, hist_scale/100))
        real_B_hist = np.histogram(real_B, bins=np.arange(0, hist_scale, hist_scale/100))
        err_B_hist = np.histogram(abs(fake_B-real_B), bins=np.arange(0, hist_scale, hist_scale/100))
        
        return real_B_hist, fake_B_hist, err_B_hist

    def calculate_accuracy(self):
        """
        """
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        predictions = (pred_fake <= 0).long()
        num_correct = predictions.sum()
        num_samples = math.prod(pred_fake.size())
        self.loss_acc_D_fake = float(num_correct) / float(num_samples)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        predictions = (pred_real > 0).long()
        num_correct = predictions.sum()
        num_samples = math.prod(pred_real.size())
        self.loss_acc_D_real = float(num_correct) / float(num_samples)
