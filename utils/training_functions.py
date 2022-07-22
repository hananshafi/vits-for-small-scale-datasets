import torch

class EarlyStopping:
    def __init__(self, patience=0, verbose=0, mode='max'):
        self._step = 0
        self._loss = 0.0
        self.patience = patience
        self.verbose = verbose
        self.best_value = 0.0
        if mode == 'max':
            self.mode = 1
        else:
            self.mode = -1 

    def validate(self, value):
        if self._loss * self.mode >= value * self.mode:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....\n\n')
                return self.best_value
        else:
            if self.best_value * self.mode < value * self.mode:
                self.best_value = value
            self._step = 0
        
        self._loss = value

        return 0
    
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res