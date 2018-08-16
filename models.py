import torch.nn as nn
from arguments import get_args

args = get_args()


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=args.disc_filters,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.disc_filters,
                      out_channels=args.disc_filters * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(args.disc_filters * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.disc_filters * 2,
                      out_channels=args.disc_filters * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(args.disc_filters * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.disc_filters * 4,
                      out_channels=args.disc_filters * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(args.disc_filters * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=args.disc_filters * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0)#,
            #nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=args.z_size,
                               out_channels=args.gener_filters * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(args.gener_filters * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=args.gener_filters * 8,
                               out_channels=args.gener_filters * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(args.gener_filters * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=args.gener_filters * 4,
                               out_channels=args.gener_filters * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(args.gener_filters * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=args.gener_filters * 2,
                               out_channels=args.gener_filters,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(args.gener_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=args.gener_filters,
                               out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


class WGAN(nn.Module):
    def __init__(self, label, z_size,
                 image_size, image_channel_size,
                 c_channel_size, g_channel_size):
        # configurations
        super().__init__()
        self.label = label
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.c_channel_size = c_channel_size
        self.g_channel_size = g_channel_size

        # components
        self.critic = Critic(
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
            channel_size=self.c_channel_size,
        )
        self.generator = Generator(
            z_size=self.z_size,
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
            channel_size=self.g_channel_size,
        )

    @property
    def name(self):
        return (
            'WGAN-GP'
            '-z{z_size}'
            '-c{c_channel_size}'
            '-g{g_channel_size}'
            '-{label}-{image_size}x{image_size}x{image_channel_size}'
        ).format(
            z_size=self.z_size,
            c_channel_size=self.c_channel_size,
            g_channel_size=self.g_channel_size,
            label=self.label,
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
        )

    def c_loss(self, x, z, return_g=False):
        g = self.generator(z)
        c_x = self.critic(x).mean()
        c_g = self.critic(g).mean()
        l = -(c_x-c_g)
        return (l, g) if return_g else l

    def g_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.critic(g).mean()
        return (l, g) if return_g else l

    def sample_image(self, size):
        return self.generator(self.sample_noise(size))

    def sample_noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.cuda() if self._is_on_cuda() else z

    def gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        a = a\
            .expand(x.size(0), x.nelement()//x.size(0))\
            .contiguous()\
            .view(
                x.size(0),
                self.image_channel_size,
                self.image_size,
                self.image_size
            )
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self._is_on_cuda() else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-(gradients+EPSILON).norm(2, dim=1))**2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
