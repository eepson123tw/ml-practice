import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel
from torchvision.utils import save_image

# 定義數據集
class TextImageDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = datasets.CIFAR10(root='./data', train=True, download=True)
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.class_descriptions = {
            0: "A cat sitting on the ground.",
            1: "A dog playing with a ball.",
            2: "A bird flying in the sky.",
            3: "A cat sitting on the ground.",
            4: "A deer standing in the forest.",
            5: "A dog running in the park.",
            6: "A frog on a lily pad.",
            7: "A horse grazing in the field.",
            8: "A ship sailing on the ocean.",
            9: "A truck driving on the road."
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        description = self.class_descriptions[label]
        tokens = self.tokenizer(description, return_tensors='pt', truncation=True, padding='max_length', max_length=16)
        if self.transform:
            image = self.transform(image)
        return image, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

# 定義生成器
class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels, feature_g):
        super(Generator, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        text_dim = self.text_encoder.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + text_dim, feature_g * 8 * 2 * 2),  # 修改為 2x2
            nn.BatchNorm1d(feature_g * 8 * 2 * 2),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),  # 2x2 -> 4x4
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.Tanh()
        )
        # 你可能需要再添加一層 ConvTranspose2d 來從 16x16 到 32x32
        self.deconv_extra = nn.Sequential(
            nn.ConvTranspose2d(img_channels, img_channels, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.Tanh()
        )

    def forward(self, noise, captions, masks):
        text_features = self.text_encoder(input_ids=captions, attention_mask=masks).pooler_output
        x = torch.cat([noise, text_features], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 2, 2)  # 修改為 2x2
        x = self.deconv(x)
        img = self.deconv_extra(x)  # 添加額外的轉置卷積層以達到 32x32
        return img

# 定義判別器
class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(Discriminator, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),  # 8x8 -> 4x4
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),  # 4x4 -> 2x2
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        text_dim = self.text_encoder.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(feature_d * 8 * 2 * 2 + text_dim, 1),  # 修改為 2x2
            nn.Sigmoid()
        )

    def forward(self, img, captions, masks):
        img_features = self.conv(img)
        img_features = img_features.view(img_features.size(0), -1)
        text_features = self.text_encoder(input_ids=captions, attention_mask=masks).pooler_output
        x = torch.cat([img_features, text_features], dim=1)
        validity = self.fc(x)
        return validity

def main():
    # 超參數設置
    noise_dim = 100
    batch_size = 64
    lr = 0.0002
    num_epochs = 50
    feature_g = 64
    feature_d = 64
    img_channels = 3

    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 數據集和數據加載器
    dataset = TextImageDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 可以調整 num_workers

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim, img_channels=img_channels, feature_g=feature_g).to(device)
    discriminator = Discriminator(img_channels, feature_d=feature_d).to(device)

    # 損失函數和優化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 標籤
    real_label = 1.
    fake_label = 0.

    # 訓練迴圈
    for epoch in range(num_epochs):
        for i, (imgs, captions, masks) in enumerate(dataloader):
            batch_size_curr = imgs.size(0)
            imgs = imgs.to(device)
            captions = captions.to(device)
            masks = masks.to(device)

            # 真實標籤
            labels_real = torch.full((batch_size_curr, 1), real_label, device=device)
            labels_fake = torch.full((batch_size_curr, 1), fake_label, device=device)

            ### 訓練判別器 ###
            discriminator.zero_grad()
            output_real = discriminator(imgs, captions, masks)
            loss_real = criterion(output_real, labels_real)
            loss_real.backward()

            # 生成假圖像
            noise = torch.randn(batch_size_curr, noise_dim, device=device)
            fake_imgs = generator(noise, captions, masks)
            output_fake = discriminator(fake_imgs.detach(), captions, masks)
            loss_fake = criterion(output_fake, labels_fake)
            loss_fake.backward()

            loss_D = loss_real + loss_fake
            optimizer_D.step()

            ### 訓練生成器 ###
            generator.zero_grad()
            output = discriminator(fake_imgs, captions, masks)
            loss_G = criterion(output, labels_real)  # 想讓生成器欺騙判別器
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} \
                      Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        # 每個epoch結束後生成一些圖像進行檢查
        with torch.no_grad():
            noise = torch.randn(16, noise_dim, device=device)
            sample_description = "A cat sitting on the ground."  # 示例文字描述
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokens = tokenizer([sample_description] * 16, return_tensors='pt', truncation=True, padding='max_length', max_length=16)
            sample_captions = tokens['input_ids'].squeeze().to(device)
            sample_masks = tokens['attention_mask'].squeeze().to(device)
            generated_imgs = generator(noise, sample_captions, sample_masks).cpu()
            # 保存生成的圖像
            save_image(generated_imgs, f"generated_epoch_{epoch+1}.png", normalize=True)

if __name__ == '__main__':
    import torch.multiprocessing
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # 如果已經設定過，則忽略
    main()
