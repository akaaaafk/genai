import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(3 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3,16,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # app
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # 2
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        #    3
        self.conv3 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 4
        self.conv4 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)


        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4*4,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        # Conf and pooling layers
        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(
            F.relu(self.bn4(self.conv4(x)))
        )  # Conv -> BatchNorm -> ReLU -> Pool

        # Flatten the feature map
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layer app with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim= 128):
        super(VAE, self).__init__()
        # width + 2p - k / s + 1
        # Encoder
        # Input(B, 3, 64,64)
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, stride = 2, padding=1)
        # Output(B,32,32,32)

        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        # Output(B,64,16,16)

        self.conv3 = nn.Conv2d(64,128,kernel_size=3, stride=2,padding=1)
        # Output(B,128,8,8)

        self.conv4 = nn.Conv2d(128,256, kernel_size=3, stride=2,padding=1)
        # output(B,256,4,4)

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # image
        #   ↓
        # shared CNN encoder
        #   ↓
        # shared feature vector x
        #   ↓              ↓
        # fc_mu        fc_logvar
        #   ↓              ↓
        # 均值           方差
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256*4*4)

        self.deconv1 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32,3,kernel_size=3,stride=2,padding=1,output_padding=1)

    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
    def reparameterize(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    def decode(self,z):
        x = self.fc_decode(z)
        x = x.view(-1,256,4,4)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        x = torch.sigmoid(self.deconv4(x))

        return x

    def forward(self,x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        x_recon = self.decode(z)

        return x_recon, mu, logvar

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # layer one
        self.conv1 = nn.Conv2d(3,16, kernel_size=3, stride = 1, padding=1)

        # layer two
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, stride=1,padding=1)

        # pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.flatten = nn.Flatten()

        # fully connected layer
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # layer app
        x = self.pool(F.relu(self.conv1(x)))
        # layer 2
        x = self.pool(F.relu(self.conv2(x)))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64,128,kernel_size=4, stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*7*7, 1)

    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim = 100):
        super(Generator, self).__init__()
        self.latent_dim = noise_dim

        self.fc1 = nn.Linear(self.latent_dim, 128*7*7)

        self.deconv1 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2, padding=1)
        self.act2 = nn.Tanh()

    def forward(self,x):
        x = self.fc1(x)
        x = x.view(x.size(0),128,7,7)
        x = self.act1(self.bn1(self.deconv1(x)))
        x = self.act2(self.deconv2(x))
        return x

class VanillaRNN(nn.Module):
    # input (B,T)
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super(VanillaRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size = hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh"
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self,x, hidden = None):
        x = self.embedding(x)

        out, hidden = self.rnn(x, hidden)

        logits = self.fc(out)

        return logits, hidden

def get_model(model_name, vocab_size=None):
    if model_name == "MLP":
        return MLP()

    elif model_name == "SimpleCNN":
        return SimpleCNN()

    elif model_name == "EnhancedCNN":
        return EnhancedCNN()

    elif model_name == "VAE":
        return VAE(latent_dim=128)

    elif model_name == "CNN":
        return CNN()

    elif model_name == "Generator":
        return Generator(noise_dim=100)

    elif model_name == "Discriminator":
        return Discriminator()

    elif model_name == "VanillaRNN":
        if vocab_size is None:
            raise ValueError("VanillaRNN requires vocab_size.")
        return VanillaRNN(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=1
        )

    else:
        raise ValueError(f"Model {model_name} not recognized.")