import argparse
import torch
import networks as nw
import numpy as np
import transformers as t
from networks import FullyConnected
from networks import AffineNet
import torch.nn as nn
from torch import optim
#import time
import itertools 

MEAN = 0.1307
SIGMA = 0.3081
DEVICE = 'cpu'
INPUT_SIZE = 28

def verify(net, inputs, eps, true_label, heuristic=None):
    verif_layers = []
    verif_layers_types = []
    # Construct the neural network verification layers
    for name, layer in net.named_modules():
        if name.startswith("layers."):
            if isinstance(layer, nw.Normalization):
                pass
            elif isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nw.SPU):
                layer_size = verif_layers[-1].get_output_size()
                spu = t.AbstractSPU(heuristic_labels=heuristic, layer_size=layer_size)
                verif_layers.append(spu)
                verif_layers_types.append('spu')
            elif isinstance(layer, nn.Linear):
                affine = t.AbstractAffine(net.state_dict()[name+".weight"].T,
                         net.state_dict()[name+".bias"].unsqueeze(1))
                verif_layers.append(affine)
                verif_layers_types.append('affine')
            else:
                pass
    # Add the last verification layer (true label minus each other label)
    diag = torch.diag(torch.ones(9)) * -1
    ones = torch.ones(9).unsqueeze(0)
    final_weights = torch.cat((diag[:true_label, :], ones, diag[true_label:, :]))
    final_biases = torch.zeros(9).unsqueeze(1)
    final_affine = t.AbstractAffine(final_weights, final_biases)
    verif_layers.append(final_affine)
    verif_layers_types.append('affine')

    # Define the neural network verifier
    net = nn.Sequential(*verif_layers)

    # Define the optimizer
    opt = optim.Adam(net.parameters(), lr=0.45)
    nb_iterations = 1500

    # Transform input to interval, clip and reshape
    input = torch.flatten(inputs)
    tmpl = torch.clip((input-eps), 0, 1).unsqueeze(1) # Proper array dimension instead of (784,) -> (784,1)
    tmpu = torch.clip((input+eps), 0, 1).unsqueeze(1)

    # Normalize the input
    tmpl -= MEAN
    tmpl /= SIGMA
    tmpu -= MEAN
    tmpu /= SIGMA

    # Construct the initial layer
    initial_layer = {
        'lowers': tmpl,
        'uppers': tmpu,
        'x_lowers': None,
        'x_uppers': None,
        'x_lowers_intercepts': tmpl,
        'x_uppers_intercepts': tmpu
    }

    # Run optimization loop
    for i in range(nb_iterations):
        opt.zero_grad()

        layers = {} # layers dict for solver
        curidx = 0 # key for layers dict, gets increased by SPU and Affine calls
        layers[curidx] = initial_layer
        for verif_layer, layer_type in zip(verif_layers, verif_layers_types):
            # Propagate the DeepPoly relaxation through the verifier network
            if layer_type == 'spu':
                layers[curidx+1] = verif_layer.forward(layers[curidx])
            elif layer_type == 'affine':
                layers[curidx+1] = verif_layer.forward(layers, curidx, max_backsubst=30)
            curidx += 1
        final_output = layers[curidx]['lowers']
        #print('lower_bound:', torch.min(final_output))

        # If all lower bounds are positive, we proved robustness
        if torch.min(final_output) > 0:
            return True

        # Else we compute the loss and run backprop
        loss = torch.log(-torch.min(final_output))
        loss.backward()
        opt.step()

    # If we did not manage to verify, either we failed or the network
    # really is not robust enough
    return False


def analyze(net, inputs, eps, true_label):
    #start_time = time.time()
    verified = verify(net, inputs, eps, true_label, heuristic=['a1', 'b8', 'c7'])
    #end_time = time.time()
    #print(f"Runtime: {end_time - start_time}")
    return verified

def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    #net flattens input from [1, 1, 28, 28]-->[1, 784]
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
