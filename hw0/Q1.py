import numpy as np
import sys
import csv
def main():
	file = open('ans_one.txt', 'w')
	with open(sys.argv[1], 'rb') as f:
		reader = csv.reader(f)
		mat_1 = list(reader)
	with open(sys.argv[2], 'rb') as f:
		reader = csv.reader(f)
		mat_2 = list(reader)
	mat_1 = np.array(mat_1, dtype='int')
	mat_2 = np.array(mat_2, dtype='int')
	result = np.dot(mat_1, mat_2)
	result = np.reshape(result, -1)
	result = sorted(result)
	for i in result :
		file.write(str(i)+'\n')
	file.close()
if __name__ == '__main__' :
	main()