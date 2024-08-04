import numpy as np
from PIL import Image
from math import log10, sqrt
import cv2
import numpy as np

e = 10000
n = 50

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main():
    original = cv2.imread("images/IUP.jpg")
    compressed = cv2.imread(f"images/denoised_image_e{e}_n{n}.jpg", 1)
    value = PSNR(original, compressed)
    print(value)


if __name__ == "__main__":


    main()