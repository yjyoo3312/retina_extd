import torch.optim as optim

def load_optimizer(net, args):
  if args.optim.lower() == 'sgd':
    return optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
  elif args.optim.lower() == 'adam':
    return optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  elif args.optim.lower() == 'adamw':
    return optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  elif args.optim.lower() == 'rmsprop':
    return optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def lr_update(epoch, opt, optimizer):
    if 0 < epoch <= opt.lr_warmup_epoch:
      mul_rate = 10 ** (1/opt.lr_warmup_epoch)
      print(">>> Learning rate warm-up : %.4f to %.4f" % (opt.lr, opt.lr * mul_rate))
      opt.lr *= mul_rate

      for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr

    elif str(epoch+1) in opt.lr_decay_epoch.split(','):
      print(">>> Learning rate decay : %.5f to %.5f" % (opt.lr, opt.lr*0.1))
      opt.lr *= 0.1

      for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr