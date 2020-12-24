import random
import math
import numpy as np

def data_pass_generator(n):
	output = []
	for i in range(n):
		center = 0
		blink = 0
		left = 0
		right = 0

		for j in range(100):
			[ rd ] = random.choices(range(0, 4), weights=[92,4,2,2])
			if rd == 0:
				center += 1
			elif rd == 1:
				blink += 1
			elif rd == 2:
				left += 1
			else:
				right += 1

		output.append([center, blink, left, right])

	return output

def data_non_pass_generator(n):
	output = []
	for i in range(n):
		center = 0
		blink = 0
		left = 0
		right = 0

		seed = random.randrange(10,30)
		for j in range(100):
			[ rd ] = random.choices(range(0, 4), weights=[65,seed,5,5])
			if rd == 0:
				center += 1
			elif rd == 1:
				blink += 1
			elif rd == 2:
				left += 1
			else:
				right += 1

		output.append([center, blink, left, right])

	return output

# Ratio <pass : non-pass> = <7 : 3>
def get_eyetracking_data(n):
	pass_cnt = math.floor(n / 100 * 70)
	non_pass_cnt = n - pass_cnt
	pass_data = data_pass_generator(pass_cnt)
	non_pass_data = data_non_pass_generator(non_pass_cnt)

	data = np.array(
		pass_data +
		non_pass_data
	)

	label = np.array(
		[
			1 for i in range(pass_cnt)
		] +
		[
			0 for i in range(non_pass_cnt)
		]
	)

	return data, label