import numpy as np
import warnings
import torch.nn as nn
import torch

sig = nn.Sigmoid() 

def print_layer(layer, verbose=True):
	print("##### START OF LAYER #####")
	for i, (l, u, x_l, x_u, c_l, c_u) in enumerate(zip(layer['lowers'], layer['uppers'], \
		layer['x_lowers'], layer['x_uppers'], layer['x_lowers_intercepts'], \
		layer['x_uppers_intercepts'])):
		expr_l = ""
		expr_u = ""
		for j, (val_l, val_u) in enumerate(zip(x_l, x_u)):
			expr_l += f"{val_l}x_{j} + "
			expr_u += f"{val_u}x_{j} + "
		if verbose:
			print(f"y_{i} >= {expr_l}{c_l}")
			print(f"y_{i} <= {expr_u}{c_u}")
		print(f"l_{i} = {l}")
		print(f"u_{i} = {u}")
		print("-----------------")
	print("##### END OF LAYER #####")


class AbstractSPU(nn.Module):

	def __init__(self, layer_size, verbose=False, heuristic_labels=None):
		super(AbstractSPU, self).__init__()
		if heuristic_labels is None:
			self.heuristic_labels = ['a1', 'b', 'c']
		else:
			self.heuristic_labels = heuristic_labels
		self.verbose = verbose
		self.lam = torch.nn.Parameter(torch.ones(layer_size) * 1.0)

		# Case u <= 0:
		heuristics_case1 = {'a1': lambda l, u: (  -self.sig_neg_l * (1-self.sig_neg_l)  ,  self.sig_neg_l * (1+l-l*self.sig_neg_l) - 1  ), 
							'a2': lambda l, u: (  -self.sig_neg_u * (1-self.sig_neg_u)  ,  self.sig_neg_u * (1+u-u*self.sig_neg_u) - 1  ),
							'a3': lambda l, u: (  -sig((u+l)/2)*(1-sig((u+l)/2))  ,  sig(-(u+l)/2) + sig((u+l)/2)*(1-sig((u+l)/2))*(u+l)/2 - 1  )}

		# Case l >= 0:
		heuristics_case2 = {'b1': lambda l, u: (  u+l  ,  (-2 - (u+l)**2)/4.  ),
							'b2': lambda l, u: (  2*u  ,  -u**2 - 0.5  ), #really bad
							'b3': lambda l, u: (  2*l  ,  -l**2 - 0.5  ),
							'b4': lambda l, u: (  2*(0.4*u+0.6*l)  , -0.5 - (0.4*u+0.6*l)**2),
							'b5': lambda l, u: (  2*(0.3*u+0.7*l)  , -0.5 - (0.3*u+0.7*l)**2),
							'b6': lambda l, u: (  2*(0.1*u+0.9*l)  , -0.5 - (0.1*u+0.9*l)**2),
							'b7': lambda l, u: (  2*(0.2*u+0.8*l)  , -0.5 - (0.2*u+0.8*l)**2),
							'b8': lambda l, u: (  2*(0.15*u+0.85*l)  , -0.5 - (0.15*u+0.85*l)**2),
							'b_param': lambda l, u: (  2*(b_p*u+(1-b_p)*l)  , -0.5 - (b_p*u+(1-b_p)*l)**2),
							'b9': lambda l, u: (  2*(l**2+u**2)/(l+u)  , -0.5 - ((l**2+u**2)**2)/((l+u)**2)  ),
							'b10': lambda l, u: (  2*(0.2*u+0.8*l)  , -0.5 - (0.2*u+0.8*l)**2  ) if (l+u)/2 < 3 else  (  2*(0.1*u+0.9*l)  , -0.5 - (0.1*u+0.9*l)**2  ),
							'b11': lambda l, u: (  2*(self.p*u+(1-self.p)*l)  , -0.5 - (self.p*u+(1-self.p)*l)**2  ),
							'b12': lambda l, u: (  2*(self.pb12*u+(1-self.pb12)*l)  , -0.5 - (self.pb12*u+(1-self.pb12)*l)**2  ),
							'b13': lambda l, u: (  2*(self.pb13*u+(1-self.pb13)*l)  , -0.5 - (self.pb13*u+(1-self.pb13)*l)**2  )
							}

		# Crossing case:
		heuristics_case3 = {'c1': lambda l, u: (  (self.sig_neg_l-0.5)/l  ,  -0.5  ), 
							'c2': lambda l, u: (  0  ,  -0.5  ),
							'c3': lambda l, u: (  2*u  ,  -u**2 - 0.5  ) if l >= -0.5 else (  (self.sig_neg_l-0.5)/l  ,  -0.5  ),
							'c4': lambda l, u: (  0  ,  -0.5  ) if abs(l) >= abs(u) else (  u+l  ,  (-2-(u+l)**2)/4.  ),
							'c5': lambda l, u: (  (u**2-self.sig_neg_l+0.5)/(u-l)  ,  -((u**2-self.sig_neg_l+0.5)/(u-l))**2/4. -0.5  ),
							'c6': lambda l, u: (  0  ,  -0.5  ) if (0.55*l+0.45*u) < 0 else (  2* (0.55*l+0.45*u)  ,  -0.5 -(0.55*l+0.45*u)**2  ),
							'c7': lambda l, u: (  0  ,  -0.5  ) if (0.7*l+0.3*u) < 0 else (  2* (0.7*l+0.3*u)  ,  -0.5 -(0.7*l+0.3*u)**2  ),
							'c8': lambda l, u: (  0  ,  -0.5  ) if (0.6*l+0.4*u) < 0 else (  2* (0.6*l+0.4*u)  ,  -0.5 -(0.6*l+0.4*u)**2  ),
							'c_param': lambda l, u: (  0  ,  -0.5  ) if (c_p*l+(1-c_p)*u) < 0 else (  2* (c_p*l+(1-c_p)*u)  ,  -0.5 -(c_p*l+(1-c_p)*u)**2  ),
							'c9': lambda l, u: (  0  ,  -0.5  ) if (0.75*l+0.25*u) < 0 else (  2* ((1-self.p2)*l+self.p2*u)  ,  -0.5 -((1-self.p2)*l+self.p2*u)**2  ),
							'c10': lambda l, u: (  0  ,  -0.5  ) if ((1-self.p2)*l+self.p2*u) < 0 else (  2* ((1-self.p2)*l+self.p2*u)  ,  -0.5 -((1-self.p2)*l+self.p2*u)**2  ),
							'c55': lambda l, u: (  (u**2-self.sig_neg_l+0.5)/(u-l)  ,  -((u**2-self.sig_neg_l+0.5)/(u-l))**2/4. -0.5  ) if u > self.u_thresh else (  0  ,  -0.5  ),
							'c11': lambda l, u: (  0  ,  -0.5  ) if ((1-self.pc11)*l+self.pc11*u) < 0 else (  2* ((1-self.pc11)*l+self.pc11*u)  ,  -0.5 -((1-self.pc11)*l+self.pc11*u)**2  )
							}
		
		self.heuristics = [heuristics_case1, heuristics_case2, heuristics_case3]

	def spu_linear_bounds(self, l, u, lam):
		self.sig_neg_l = sig(-l)
		self.sig_neg_u = sig(-u)

		# linear bounds of the form: m * x + c
		sl = sig(lam)
		if u <= 0:
			# Lower bound
			m_x_lower = (self.sig_neg_u - self.sig_neg_l)/(u-l)
			c_x_lower = (self.sig_neg_l * u - self.sig_neg_u * l)/(u-l) - 1

			# Upper bound
			m_x_upper, c_x_upper = self.heuristics[0][self.heuristic_labels[0]](l, u)

		elif l >= 0:
			# Lower bound
			m_x_lower = 2*(sl*u+(1-sl)*l)
			c_x_lower = -0.5 - (sl*u+(1-sl)*l)**2

			# Upper bound
			m_x_upper = u+l
			c_x_upper = -l*u - 0.5

		else: # crossing case
			u_thresh = (-self.sig_neg_l * (1-self.sig_neg_l) + torch.sqrt((self.sig_neg_l * (1-self.sig_neg_l))**2 + 4*(self.sig_neg_l * (1+l-l*self.sig_neg_l) - 0.5)))/2.
			self.u_thresh = u_thresh

			# Lower bound
			ll = (self.sig_neg_l - 0.5)/(2*l)
			m_x_lower = 2*(sl*u+(1-sl)*ll)
			c_x_lower = -0.5 - (sl*u+(1-sl)*ll)**2

			# Upper bound
			if u > u_thresh:
				m_x_upper = (u**2-self.sig_neg_l+0.5)/(u-l)
				c_x_upper = u**2 - (u**2-self.sig_neg_l+0.5)/(u-l)*u - 0.5
			else:
				m_x_upper = -self.sig_neg_l*(1-self.sig_neg_l)
				c_x_upper = self.sig_neg_l*(1+l-l*self.sig_neg_l) - 1

		return m_x_lower, c_x_lower, m_x_upper, c_x_upper

	def forward(self, layer):
		"""
		Performs the abstract transformation of SPU layer

		Parameters
		----------
		layer: a dictionary that stores all values of the layer, of the form:
			layer = {lowers: 				Lx1,
					 uppers: 				Lx1,
					 x_lowers: 				LxL_prev,
					 x_uppers: 				LxL_prev,
					 x_lowers_intercepts: 	Lx1,
					 x_uppers_intercepts: 	Lx1}

		Returns
		-------
		layer : dictionary of all the arrays (lowers, x_lowers, ...)
			A transformed layer with the same shape as the original layer
		"""
		if self.verbose:
			print("\tSPU unit - performing forward pass")

		lowers = torch.zeros_like(layer['lowers'])
		uppers = torch.zeros_like(layer['uppers'])
		x_lowers = None
		x_uppers = None
		x_lowers_intercepts = torch.zeros_like(layer['x_lowers_intercepts'])
		x_uppers_intercepts = torch.zeros_like(layer['x_uppers_intercepts'])

		for i, (l, u) in enumerate(zip(layer['lowers'], layer['uppers'])):

			new_x_lower = torch.zeros_like(layer['lowers']).flatten()
			new_x_upper = torch.zeros_like(layer['uppers']).flatten()

			m_x_lower, c_x_lower, m_x_upper, c_x_upper = self.spu_linear_bounds(l, u, self.lam[i])

			new_x_lower[i] = m_x_lower
			new_x_upper[i] = m_x_upper

			#evaluate at l or u, whichever yields smaller value (new_l) or largest value (new_u)
			new_l = min(new_x_lower[i] * l, new_x_lower[i] * u) + c_x_lower
			new_u = max(new_x_upper[i] * l, new_x_upper[i] * u) + c_x_upper
			
			#This throws exception in some case, but seems to be a rounding error
			#assert new_l <= new_u, f"error: (new_l, new_u) = ({new_l}, {new_u})"

			lowers[i] = new_l
			uppers[i] = new_u
			if x_lowers is None:
				x_lowers = new_x_lower
			else:
				x_lowers = torch.vstack((x_lowers, new_x_lower))
			if x_uppers is None:
				x_uppers = new_x_upper
			else:
				x_uppers = torch.vstack((x_uppers, new_x_upper))
			x_lowers_intercepts[i] = c_x_lower
			x_uppers_intercepts[i] = c_x_upper

		"""
		assert lowers.shape == layer['lowers'].shape
		assert uppers.shape == layer['uppers'].shape
		assert x_lowers.shape[0] == x_lowers.shape[1]
		assert x_uppers.shape[0] == x_uppers.shape[1]
		assert x_lowers_intercepts.shape == layer['x_lowers_intercepts'].shape
		assert x_uppers_intercepts.shape == layer['x_uppers_intercepts'].shape
		"""

		layer = {'lowers': lowers,
				 'uppers': uppers,
				 'x_lowers': x_lowers,
				 'x_uppers': x_uppers,
				 'x_lowers_intercepts': x_lowers_intercepts,
				 'x_uppers_intercepts': x_uppers_intercepts}

		if self.verbose:
			print("\tSPU unit - finished forward pass")
			print_layer(layer)

		return layer


class AbstractAffine(nn.Module):

	def __init__(self, weights, biases, verbose=False):
		super(AbstractAffine, self).__init__()
		self.weights = weights # NxD
		self.biases = biases # Dx1
		self.verbose = verbose

	def get_output_size(self):
		return self.biases.shape[0]

	def forward(self, network, layer_idx, max_backsubst=30):
		"""
		Performs the abstract transformation of the affine layer, with backsubstitution

		Parameters
		----------
		network: A dictionary where the keys represent the layer_idx
				 i.e. network = {0: layer_dictionary, 1: layer_dictionary, 2: ...})
			stores all layers (stores every single value of the network)
		layer_idx: 		int
			index of current layer in network, takes values in [0, ..., #layers-1]
		max_backsubst: 	int
			maximum number of backsubstitutions to perform

		Returns
		-------
		layer : dictionary of all the arrays (lowers, x_lowers, ...)
		"""

		if self.verbose:
			print("\tAffine unit - performing forward pass")

		layer = network[layer_idx]

		x_lowers = self.weights.T # DxN
		x_uppers = self.weights.T
		x_lowers_intercepts = self.biases # Dx1
		x_uppers_intercepts = self.biases

		"""
		assert len(x_lowers_intercepts.shape) == 2, "x_lowers_intercepts must be 2D" # Must be (D,1) not (D,)
		assert x_lowers_intercepts.shape[1] == 1
		assert x_lowers_intercepts.shape[0] == x_lowers.shape[0] #first dim of x_lowers and x_lowers_intercepts must be identitcal
		"""

		# Back-substitution
		if self.verbose:
			print(f"Performing back-substitution - layer index : {layer_idx}")
		_x_lowers = x_lowers.clone()
		_x_uppers = x_uppers.clone()
		_x_lowers_intercepts = x_lowers_intercepts.clone()
		_x_uppers_intercepts = x_uppers_intercepts.clone()
		for i in range(0, min(max_backsubst, layer_idx)): # nb of back-substitutions			
			"""
			For the 'lowers' case for example, if the coef is positive we replace with the lower bound
			else if the coef is negative we replace with the upper bound.
			torch.maximum() and torch.minimum() allow the separation of these 2 cases
			"""
			_x_lowers_intercepts += torch.maximum(_x_lowers, torch.zeros_like(_x_lowers)) @ network[layer_idx-i]['x_lowers_intercepts'] + \
				torch.minimum(_x_lowers, torch.zeros_like(_x_lowers)) @ network[layer_idx-i]['x_uppers_intercepts']
			_x_uppers_intercepts += torch.maximum(_x_uppers, torch.zeros_like(_x_uppers)) @ network[layer_idx-i]['x_uppers_intercepts'] + \
				torch.minimum(_x_uppers, torch.zeros_like(_x_uppers)) @ network[layer_idx-i]['x_lowers_intercepts']

			_x_lowers = torch.maximum(_x_lowers, torch.zeros_like(_x_lowers)) @ network[layer_idx-i]['x_lowers'] + \
				torch.minimum(_x_lowers, torch.zeros_like(_x_lowers)) @ network[layer_idx-i]['x_uppers']
			_x_uppers = torch.maximum(_x_uppers, torch.zeros_like(_x_uppers)) @ network[layer_idx-i]['x_uppers'] + \
				torch.minimum(_x_uppers, torch.zeros_like(_x_uppers)) @ network[layer_idx-i]['x_lowers']

		lower_prod = torch.maximum(_x_lowers, torch.zeros_like(_x_lowers)) * network[0]['lowers'].T + \
			torch.minimum(_x_lowers, torch.zeros_like(_x_lowers)) * network[0]['uppers'].T
		lowers = torch.sum(lower_prod, dim=1, keepdim=True) + _x_lowers_intercepts
		upper_prod = torch.maximum(_x_uppers, torch.zeros_like(_x_uppers)) * network[0]['uppers'].T + \
			torch.minimum(_x_uppers, torch.zeros_like(_x_uppers)) * network[0]['lowers'].T
		uppers = torch.sum(upper_prod, dim=1, keepdim=True) + _x_uppers_intercepts

		#assert torch.all(lowers <= uppers), print(uppers - lowers)

		layer = {'lowers': lowers,
				 'uppers': uppers,
				 'x_lowers': x_lowers,
				 'x_uppers': x_uppers,
				 'x_lowers_intercepts': x_lowers_intercepts,
				 'x_uppers_intercepts': x_uppers_intercepts}

		if self.verbose:
			print("\tAffine unit - finished forward pass")
			print_layer(layer)

		return layer