import os

import os

def main():
	i = 0
	path="C:/Users/mined/Downloads/data/imagesTs/"
	for filename in os.listdir(path):
		my_dest ="img00" + str(i) + ".nii.gz"
		my_source = path + filename
		my_dest = path + my_dest

		os.rename(my_source, my_dest)
		i += 1

if __name__ == '__main__':
	main()