from PIL import Image
import sys
def main():
	img_1 = Image.open(sys.argv[1])
	img_2 = Image.open(sys.argv[2])
	width, heigth = img_1.size
	pix_1 = img_1.load()
	pix_2 = img_2.load()
	for i in range(0, width) :
		for j in range(0, heigth) :
			if pix_1[i, j] == pix_2[i, j] :
				pix_2[i, j] = (0, 0, 0, 0)
	img_2.save("ans_two.png")
if __name__ == '__main__':
	main()