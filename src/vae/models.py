import torch
import torch.nn as nn


class LeNeT5Classifier(nn.Module):
    def __init__(self, latent_dim: int = 120):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout1d(p=0.2)
        self.sigmoid = nn.Tanh()
        self.linear1 = nn.Linear(self.latent_dim, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        return x


class LeNeT5Encoder(nn.Module):
    def __init__(self, latent_dim: int = 120):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv11 = nn.Conv2d(
            1, 6, kernel_size=3, padding=2,
        )
        self.conv12 = nn.Conv2d(
            6, 6, kernel_size=3, padding=0,
        )

        self.conv21 = nn.Conv2d(
            6, 16, kernel_size=3, padding=0,
        )

        self.conv22 = nn.Conv2d(
            16, 16, kernel_size=3, padding=0,
        )

        #self.bn1 = nn.BatchNorm2d(num_features=6)
        #self.bn2 = nn.BatchNorm2d(num_features=16)
        #self.dropout = nn.Dropout2d(p=0.2)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.sigmoid = nn.Tanh()
        self.linear = nn.Linear(400, self.latent_dim)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv11(x)
        #x = self.bn1(x)
        x = self.act(x)
        #x = self.dropout(x)
        x = self.conv12(x)
        #x = self.bn2(x)
        x = self.act(x)
        #x = self.dropout(x)
        x = self.pool(x)

        x = self.conv21(x)
        #x = self.bn2(x)
        x = self.act(x)
        #x = self.dropout(x)
        x = self.conv22(x)
        #x = self.bn2(x)
        x = self.act(x)
        #x = self.dropout(x)
        x = self.pool(x)

        x = x.flatten(start_dim=1,)

        x = self.linear(x)
        x = self.sigmoid(x)


        return x


class LeNeT5Decoder(nn.Module):
    def __init__(self, latent_dim: int = 120):
        super().__init__()
        self.latent_dim = latent_dim
        self.upsample = nn.Upsample(
            scale_factor=2, mode='nearest'
        )
        
        self.conv11 = nn.ConvTranspose2d(
            16, 16, kernel_size=3, padding=0,
        )
        self.conv12 = nn.ConvTranspose2d(
            16, 6, kernel_size=3, padding=0,
        )

        self.conv21 = nn.ConvTranspose2d(
            6, 6, kernel_size=3, padding=0,
        )

        self.conv22 = nn.ConvTranspose2d(
            6, 1, kernel_size=3, padding=2,
        )

        self.act = nn.ReLU()
        self.sigmoid = nn.Tanh()
        self.linear = nn.Linear(self.latent_dim, 400)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.sigmoid(x)
        x = x.reshape(-1, 16, 5, 5)

        x = self.upsample(x)

        x = self.conv11(x)
        x = self.act(x)
        x = self.conv12(x)
        x = self.act(x)


        x = self.upsample(x)
        x = self.conv21(x)
        x = self.act(x)
        x = self.conv22(x)

    
        return x


class SamplerMixin:
    def sample_from_latents(self, means: torch.Tensor = 0, stds: torch.Tensor = 1) -> torch.Tensor:
        pass


class AE(nn.Module, SamplerMixin):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert self.encoder.latent_dim == self.decoder.latent_dim
        self.latent_dim = self.encoder.latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def sample_from_latent(self, means: torch.Tensor = 0, stds: torch.Tensor = 1) -> torch.Tensor:
        z = means + stds * torch.randn(means.shape)
        x = self.decoder(z)
        return x


class VAE(nn.Module, SamplerMixin):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert self.encoder.latent_dim == self.decoder.latent_dim
        self.latent_dim = self.encoder.latent_dim

        self.means = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_vars = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        means    = self.means(x)
        log_vars = self.log_vars(x)
        z = self.reparametrization_trick(means, log_vars)
        x = self.decoder(z)

        return x, (means, log_vars)

    def reparametrization_trick(self, means, log_vars):
        eps = torch.randn(means.shape)
        stds = (0.5*log_vars).exp()
        assert (stds >= -0.0).all(), f"{stds[stds < 0]}"

        return means + stds * eps

    def sample_from_latent(self, means: torch.Tensor = 0, stds: torch.Tensor = 1) -> torch.Tensor:
        z = means + stds * torch.randn(means.shape)
        x = self.decoder(z)
        return x