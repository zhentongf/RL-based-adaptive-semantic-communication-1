import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn

# from CIFAR import compute_composite_snr_db, googlenet
# from fuzzy_logic import decide_use_nn
def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    """Helper function to create a conv-BN-ReLU block"""
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer
    
class inception(nn.Module):
    """Inception module as used in GoogleNet"""
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # Four parallel pathways
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)  # 1x1 conv
        
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)  # 1x1 followed by 3x3
        )
        
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)  # 1x1 followed by 5x5
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)  # Pooling followed by 1x1
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)  # Concatenate all branches
        return output

class googlenet(nn.Module):
    """GoogleNet architecture for CIFAR-10 classification"""
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose
        
        # Define the network blocks
        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )
        
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through each block
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.classifier(x)  # Final classification layer
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Load classifier
# =========================
classifier = googlenet(3, 10)
classifier.load_state_dict(torch.load('saved_models/google_net.pkl'))
classifier.to(device)
classifier.eval()

# =========================
# CIFAR-10 Test Set
# =========================
def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    x = x.transpose((2, 0, 1))
    return torch.from_numpy(x)

test_set = datasets.CIFAR10('./datasets/cifar10', train=False, transform=data_tf, download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# =========================
# Load simulation data
# =========================
df = pd.read_csv("nearest_cars_data.csv")

# =========================
# Models to test
# =========================
model_snr_list = [10.0, 15.0, 17.5, 20.0, 25.0]

# =========================
# RED-CNN (same as training)
# =========================
class RED_CNN(nn.Module):
    def __init__(self, dimension=576):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2)
        self.conv3 = nn.Conv2d(128, 256, 4, 1)
        self.conv4 = nn.Conv2d(256, 384, 5, 1)
        self.conv5 = nn.Conv2d(384, 512, 5, 1)
        self.conv6 = nn.Conv2d(512, dimension, 3, 1)

        self.tconv1 = nn.ConvTranspose2d(dimension, 512, 3, 1)
        self.tconv2 = nn.ConvTranspose2d(512, 384, 5, 1)
        self.tconv3 = nn.ConvTranspose2d(384, 256, 5, 1)
        self.tconv4 = nn.ConvTranspose2d(256, 128, 5, 1)
        self.tconv5 = nn.ConvTranspose2d(128, 64, 5, 2)
        self.tconv6 = nn.ConvTranspose2d(64, 3, 4, 2)

    def forward(self, x, snr):
        import torch.nn.functional as F
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))

        # Noise (semantic channel)
        out_np = out.detach().cpu().numpy()
        power = np.mean(out_np ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = torch.randn_like(out) * np.sqrt(noise_power)
        out = out + noise.to(device)

        out = F.relu(self.tconv1(out))
        out = F.relu(self.tconv2(out))
        out = F.relu(self.tconv3(out))
        out = F.relu(self.tconv4(out))
        out = F.relu(self.tconv5(out))
        out = F.relu(self.tconv6(out))
        return out

# =========================
# Metrics
# =========================
def compute_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return 10 * torch.log10(1.0 / mse)

# =========================
# Create result folder
# =========================
os.makedirs("./results/transmission_cifar", exist_ok=True)

results = []

# =========================
# MAIN LOOP
# =========================
for model_snr in model_snr_list:

    print(f"\n=== Testing Model SNR {model_snr} ===")

    model_path = f"./saved_models/CIFAR_encoder_1.000000_snr_{model_snr:.2f}.pkl"
    compression_rate = 1.0
    dimension = int(96 * 96 * 3 * compression_rate / (8 * 8))

    encoder = RED_CNN(dimension=dimension).to(device)
    encoder.load_state_dict(torch.load(model_path))
    encoder.eval()

    for idx, row in df.iterrows():

        use_nn = row["use_nn"]
        composite_snr = row["composite_snr_db"]

        total_acc = 0
        total_psnr = 0
        count = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_nn:
                # Semantic
                outputs = encoder(images, model_snr)
            else:
                # Direct channel
                img_np = images.detach().cpu().numpy()
                power = np.mean(img_np ** 2)
                noise_power = power / (10 ** (composite_snr / 10))
                noise = torch.randn_like(images) * np.sqrt(noise_power)
                outputs = images + noise.to(device)

            # Classification
            preds = classifier(outputs)
            _, predicted = preds.max(1)
            acc = (predicted == labels).float().mean()

            # PSNR
            psnr = compute_psnr(outputs, images)

            total_acc += acc.item()
            total_psnr += psnr.item()
            count += 1

        avg_acc = total_acc / count
        avg_psnr = total_psnr / count

        result_row = row.to_dict()
        result_row["model_snr"] = model_snr
        result_row["accuracy"] = avg_acc
        result_row["psnr"] = avg_psnr

        results.append(result_row)

        print(f"Row {idx} | Acc={avg_acc:.4f}, PSNR={avg_psnr:.2f}")

# =========================
# Save results
# =========================
df_out = pd.DataFrame(results)
df_out.to_csv("./results/transmission_cifar/transmission_cifar_data.csv", index=False)

print("\n✅ Experiment completed and saved!")