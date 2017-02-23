import sys
from PIL import Image
im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])
width,height = im2.size
im = Image.new("RGBA",(width,height),(0,0,0,0))
for x in range(width):
	for y in range(height):
		if im2.getpixel((x,y))!=im1.getpixel((x,y)) :
			im.putpixel(((x,y)),im2.getpixel((x,y)))

im.save("ans_two.png")
