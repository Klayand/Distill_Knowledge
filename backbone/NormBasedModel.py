import torch
from torchvision import transforms


class CIFARNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform=transforms.Compose([transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])]),
        **kwargs
    ):
        super(CIFARNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class ImageNetNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        **kwargs
    ):
        super(ImageNetNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class MnistNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self, model: torch.nn.Module, transform=transforms.Compose([transforms.Normalize((0.5,), (0.5,))]), **kwargs
    ):
        super(MnistNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class MnistmNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self, model: torch.nn.Module, transform=transforms.Compose([transforms.Normalize((0.5,), (0.5,))]), **kwargs
    ):
        super(MnistmNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class PacsNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        **kwargs
    ):
        super(PacsNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class SvhnNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self, model: torch.nn.Module, transform=transforms.Compose([transforms.Normalize((0.5,), (0.5,))]), **kwargs
    ):
        super(SvhnNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class UspsNormModel(torch.nn.Module):
    """
    **kwargs aims to unify with "pretrained=True"
    """

    def __init__(
        self, model: torch.nn.Module, transform=transforms.Compose([transforms.Normalize((0.5,), (0.5,))]), **kwargs
    ):
        super(UspsNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)
