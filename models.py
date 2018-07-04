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
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
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
