import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import re

def Gan(noisy_file, steps):
    def extract_numbers_from_string(input_string):
        # Define the regular expression pattern to match numbers
        pattern = r'\d+'

        # Use re.findall to find all occurrences of the pattern in the input string
        numbers_list = re.findall(pattern, input_string)

        # Convert the list of strings to a list of integers (if needed)
        numbers_list = list(map(int, numbers_list))

        return numbers_list

    # Set the environment variable to avoid the OpenMP warning
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Generator model for denoising (deeper model with more layers)
    start_time = time.time()
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # Prepare data and model
    image_shape = (3, 737, 499)

    # Load original and noisy images
    original_image = Image.open('images/IUP.jpg').convert("RGB").resize(image_shape[1:], Image.BILINEAR)
    noisy_image = Image.open(noisy_file).convert("RGB").resize(image_shape[1:], Image.BILINEAR)

    noisiness = extract_numbers_from_string(noisy_file)

    # Convert images to tensors
    to_tensor = ToTensor()
    original_image_tensor = to_tensor(original_image).unsqueeze(0)
    noisy_image_tensor = to_tensor(noisy_image).unsqueeze(0)

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_image_tensor = original_image_tensor.to(device)
    noisy_image_tensor = noisy_image_tensor.to(device)

    # Build the generator model
    generator = Generator().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=0.001)

    # Training GAN
    epochs = steps
    batch_size = 1

    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_loss = []
        for i in tqdm(range(0, noisy_image_tensor.size(0), batch_size), desc="Batch", leave=False):
            noisy_batch = noisy_image_tensor[i:i+batch_size]
            original_batch = original_image_tensor[i:i+batch_size]

            optimizer.zero_grad()
            denoised_batch = generator(noisy_batch)
            loss = criterion(denoised_batch, original_batch)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        epoch_loss = np.mean(epoch_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | GAN Loss: {epoch_loss:.4f}")

    # Denoise the noisy image
    with torch.no_grad():
        denoised_image_tensor = generator(noisy_image_tensor)

    # Move denoised tensor back to CPU and convert to numpy array
    denoised_image_array = denoised_image_tensor.cpu().numpy()[0]

    # Save denoised image as a PNG file
    denoised_image_pil = Image.fromarray((denoised_image_array.transpose(1, 2, 0) * 255).astype(np.uint8))
    denoised_image_pil.save(f'images/denoised_image_e{epochs}_n{noisiness[0]}.jpg')

    # Save the generator model
    torch.save(generator.state_dict(), f'data/generator_model_e{epochs}_n{noisiness[0]}.pth')

    # Calculate total training time
    end_time = time.time()
    total_time = end_time - start_time

    # Save stats to a text file
    with open(f'data/training_stats_e{epochs}_n{noisiness[0]}.txt', 'w') as f:
        f.write(f"Total Training Time: {total_time:.2f} seconds\n")
        f.write(f"Number of Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Noisiness: {noisiness}\n")
        f.write(f"Generator Model File: generator_model.pth\n")
        f.write(f"Denoised Image File: denoised_image.jpg\n")

    # Plot the original, noisy, and denoised images using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image)
    plt.title("Noisy Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image_array.transpose(1, 2, 0))
    plt.title("Denoised Image")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'data/training_stats_e{epochs}_n{noisiness[0]}.png')
    # plt.show()


if __name__=="__main__":
    noisy_images = ['images/IUP_noisy_10.jpg', 'images/IUP_noisy_25.jpg', 'images/IUP_noisy_50.jpg']
    epoch_steps = [1000, 10000]

    for j in epoch_steps:
        for i in noisy_images:
            Gan(i,j)