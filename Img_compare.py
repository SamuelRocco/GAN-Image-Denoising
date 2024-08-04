# import module
from PIL import Image, ImageChops, ImageStat
e = 10000
n = 50
# assign images
img1 = Image.open("images/IUP.jpg")
img2 = Image.open(f"images/denoised_image_e{e}_n{n}.jpg")

# finding difference
diff = ImageChops.difference(img1, img2)
stat = ImageStat.Stat(diff)
diff_ratio = sum(stat.mean)/(len(stat.mean)*255)

# showing the difference
print(abs((diff_ratio * 100)-100))
diff.show()

diff.save(f"images/denoised_e{e}_n{n}_comparison.jpg")