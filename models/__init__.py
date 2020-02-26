import time
import torch
import sys
import os
import torch.nn.functional as F
from models.retinaface import RetinaFace
from data import *

class Stopwatch:
    def __init__(self, title, silance=True):
        self.title = title
        self.silance = silance

    def __enter__(self):
        self.t0 = time.time()
        #logging.debug('{} begin'.format(self.title))

    def __exit__(self, type, value, traceback):
        current_time = time.time()
        if not self.silance:
            print('{} : {}ms'.format(self.title, int((current_time - self.t0) * 1000)))
        self.latency = current_time - self.t0

class DummyModule(torch.nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

class Identity(torch.nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1, groups=1, static_padding=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1, groups=1, **kwargs)

        self.static_padding = Identity if static_padding is None else static_padding
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=dilation, groups=groups
                         )

    def forward(self, x):
        x = self.static_padding(x)
        x = self.conv(x)
        return x

import torch
import torch.nn as nn

def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2
        
        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
            
        return conv
        
    else:
        return False
    
def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
            
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model

def check_latency(net, c_in=3, s_size_h=640, s_size_w=640, bn_fold=True, repeat=500, replace_denormals=False):
    if bn_fold:
        x = torch.rand(
                1,
                c_in,
                s_size_h,
                s_size_w,
                requires_grad=False
            )
        
        with torch.no_grad():
            bn_out = net(x)
        fuse_bn_recursively(net)

        with torch.no_grad():
            bn_fold_out = net(x)
        
        print("BN Folded... (Difference : %f)" % torch.mean(torch.sqrt(torch.pow(bn_out[0] - bn_fold_out[0], 2))))
    
    if replace_denormals:
        ReplaceDenormals(net)

    torch.set_grad_enabled(False)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_num_threads(1)
    print('python version: %s' % sys.version)
    print('torch.__version__:%s' % torch.__version__)
    print('torch.backends.mkl.is_available(): %s' % torch.backends.mkl.is_available())
    print('torch.backends.openmp.is_available(): %s' % torch.backends.openmp.is_available())
    print(os.popen('conda list mkl').read())
    print('num_threads: %d' % torch.get_num_threads())

    batch_size = 1

    warm_start = 10
    repeat_count = 0

    timer = Stopwatch('latency', silance=True)

    elapsed = 0.
    for it in range(repeat + warm_start + 1):
        with torch.no_grad():
            x = torch.rand(
                batch_size,
                c_in,
                s_size_h,
                s_size_w,
                requires_grad=False
            )

        with timer:
            out = net(x)
        
        if it > warm_start:
            elapsed += timer.latency
            repeat_count += 1
            if it % 10 == 0:
                print('trial: %d, latency %f' % (repeat_count, timer.latency))

    print('elapsed: %f' % (elapsed/repeat_count))

def resume_net_and_cuda(net, args):
    if args.resume_net.endswith('.pth'):
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

        print("Successfully loaded trained weights from %s" % args.resume_net)

    if torch.cuda.device_count() > 1:
        print("Multi-gpu Training (%d GPUs)" % torch.cuda.device_count())
        net = torch.torch.nn.DataParallel(net).cuda()
    elif torch.cuda.is_available():
        net = net.cuda()
    
    return net

def ReplaceDenormals(net, thresh=1e-30, print_log=True):
    """Preventing learned parameters from being subnormal(denormal) numbers in floating-point representation
    """
    
    if print_log:
        print('Start to detect denormals in the trained-model, please wait for a while')
        total_denormals_count = 0
        total_normals_count = 0
        
    net = net.cpu()
    for name, param in net.named_parameters():
        
        if print_log:
            param_array = param.data.numpy()
            n_denormals = len(np.where((np.abs(param_array) < thresh) & (param_array != 0.0))[0])
            n_normals   = np.size(param_array) - n_denormals
            
            total_denormals_count += n_denormals
            total_normals_count   += n_normals
            
        param_denormed = torch.where(torch.abs(param) < thresh,
                                     torch.Tensor([0]),
                                     param)
        param.data.copy_(param_denormed.data)
    
    if print_log:
        total = total_normals_count + total_denormals_count
        print('All params: %d, normals: %d, denormals: %d, ratio: %f' % \
                (total, total_normals_count, total_denormals_count, total_denormals_count / total * 1.0))

def load_network(args, phase='train'):
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50-3fpn":
        cfg = cfg_re50_3fpn
    elif args.network == "resnet50-5fpn":
        cfg = cfg_re50_5fpn
    elif args.network == 'efficientb0-5fpn':
        cfg = cfg_efcb0_5fpn
    elif args.network == 'efficientb0-3fpn':
        cfg = cfg_efcb0_3fpn
    elif args.network == 'efficientb1-3fpn':
        cfg = cfg_efcb1_3fpn
    elif args.network == 'efficientb2-3fpn':
        cfg = cfg_efcb2_3fpn
    elif args.network == 'efficientb3-3fpn':
        cfg = cfg_efcb3_3fpn
    elif args.network == 'efficientb4-3fpn':
        cfg = cfg_efcb4_3fpn
    else:
        print("Please check network name argument again.")
        exit(0)

    return RetinaFace(args, cfg=cfg, phase=phase), cfg