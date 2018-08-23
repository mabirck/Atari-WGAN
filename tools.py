from torch.nn import init

def gaussian_intiailize(model, std=.01):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters()
    ]

    for p in parameters:
        if p.dim() >= 2:
            init.normal(p, std=std)
        else:
            init.constant(p, 0)
