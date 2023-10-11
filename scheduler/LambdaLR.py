class LambdaLR:
    """
    Proposer: Zikai Zhou
    """

    def __init__(
            self,
            optimizer,
    ):
        self.optimizer = optimizer

    def step(self, epoch):

        if epoch < 300:
            now_lr = 0.1
        elif 300 < epoch < 500:
            now_lr = 0.04
        else:
            now_lr = 0.01

        for group in self.optimizer.param_groups:
            group['lr'] = now_lr

        print(f"now lr = {now_lr}")


class Lambda_EMD:
    def __init__(
            self,
            optimizer,
    ):
        self.optimizer = optimizer

    def step(self, epoch):

        if epoch < 50:
            now_lr = 0.1
        elif 50 < epoch < 100:
            now_lr = 0.04
        else:
            now_lr = 0.01

        for group in self.optimizer.param_groups:
            group['lr'] = now_lr

        print(f"now lr = {now_lr}")