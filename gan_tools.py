import torch
import numpy as np
from torch import autograd
from arguments import get_args

args = get_args()



def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size, real_data.nelement()/args.batch_size).contiguous().view(args.batch_size, 3, 32, 32)
    alpha = alpha.cuda(gpu) if not args.no_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if not args.no_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if not args.no_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in xrange(10):
        samples_100 = torch.randn(100, 128)
        if not args.no_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))
