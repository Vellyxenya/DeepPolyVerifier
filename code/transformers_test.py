from transformers import AbstractSPU, sigmoid
import numpy as np

PRINT_EVERY = 100

abstractSPU = AbstractSPU()

def spu(x):
	return x**2 - 0.5 if x >= 0 else sigmoid(-x) - 1

for i in range(5000):
	#l, u = np.random.randn(), np.random.randn()
	l, u = np.random.uniform(-6,6), np.random.uniform(-6,6)
	if l > u:
		l, u = u, l
	w_l, b_l, w_u, b_u = abstractSPU.spu_linear_bounds(l, u)

	if i % PRINT_EVERY == 0:
		print(f"test case: {i}")

	for x in np.linspace(l, u, 1000):
		assert spu(x) > w_l * x + b_l - 1e-5, print('1: l, u:', l, u, spu(x), w_l * x + b_l - 1e-5)
	for x in np.linspace(l, u, 1000):
		assert spu(x) < w_u * x + b_u + 1e-5, print('2: l, u:', l, u, spu(x), w_u * x + b_u + 1e-5)

	print("SPU is sound")