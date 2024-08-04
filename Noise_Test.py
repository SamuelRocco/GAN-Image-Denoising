from wand.image import Image

# Read image using Image() function
with Image(filename="images/IUP.jpg") as img:
    atten = 50
    # Generate noise image using noise() function
    img.noise("laplacian", attenuate=atten)
    img.save(filename=f"images/IUP_noisy_{atten}.jpg")