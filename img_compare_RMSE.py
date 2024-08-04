import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def main():
    image1_path = "images/IUP.jpg"
    image2_path = "images/denoised_image_e10000_n50.jpg"

    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    psnr = calculate_psnr(img1, img2)
    rmse = calculate_rmse(img1, img2)

    print("PSNR:", psnr)
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()
