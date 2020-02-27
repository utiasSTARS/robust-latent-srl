class ExponentialScheduler():
    def __init__(self, opt, lr_decay=1, lr_decay_frequency=1, min_lr=3e-6):
        self._opt = opt
        self._lr_decay = lr_decay
        self._lr_decay_frequency = lr_decay_frequency
        self._min_lr = min_lr
        self._epoch = 0

    def __call__(self):
        if (self._epoch + 1) % self._lr_decay_frequency == 0:
            for param_group in self._opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self._lr_decay, self._min_lr)
        self._epoch += 1

    def step(self):
        self()
