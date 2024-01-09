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


class Lambda_ImageNet:
    def __init__(
            self,
            optimizer,
    ):
        self.optimizer = optimizer

    def step(self, epoch):

        if epoch < 30:
            now_lr = 0.1
        elif 30 < epoch < 60:
            now_lr = 0.01
        elif 60 < epoch < 90:
            now_lr = 0.001
        else:
            now_lr = 0.001

        for group in self.optimizer.param_groups:
            group['lr'] = now_lr

        print(f"now lr = {now_lr}")


class Lambda_Cifar_cka:
    def __init__(
            self,
            optimizer,
    ):
        self.optimizer = optimizer
        self.init_lr = 0.05

    def step(self, epoch):

        if 150 <= epoch <= 240:
            if epoch % 30 == 0:
                self.init_lr /= 10

        for group in self.optimizer.param_groups:
            group['lr'] = self.init_lr

        print(f"now lr = {self.init_lr}")


class Lambda_ImageNet_cka:
    def __init__(
            self,
            optimizer,
    ):
        self.optimizer = optimizer
        self.init_lr = 0.1

    def step(self, epoch):

        if 1 < epoch <= 100:
            if epoch % 25 == 0:
                self.init_lr /= 5

        for group in self.optimizer.param_groups:
            group['lr'] = self.init_lr

        print(f"now lr = {self.init_lr}")