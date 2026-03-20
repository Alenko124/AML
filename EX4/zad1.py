import torch

def c(t):
    _t = t.reshape(-1, 1)
    return torch.concat((2*_t + 1, -_t**2), dim=1)

def speed(t):
    return 2 * torch.sqrt(1 + t**2)

def curve_length(c, T):
    ct = c(T)  # |T| x 2
    delta = ct[1:] - ct[:-1]  # (|T|-1) x 2
    retval = torch.sum(delta.reshape(T.numel() - 1, -1)**2, dim=1).sqrt().sum()
    return retval

def integrate_speed(T):
    s = speed(T)  # |T|
    dt = T[1] - T[0]  # assuming a linspace
    return s.sum() * dt

print('The curve length is {}'.format(curve_length(c, torch.linspace(0, 1, 100))))
print('The integrated speed is {}'.format(integrate_speed(torch.linspace(0, 1, 100))))