# The purpose of this script is to check if one can train only the INITIAL
# layers of a network, while keeping the FINAL layers frozen. This is
# accomplished by performing full backprop, but only updating the weights for
# the initial layers.
#
# I believe this makes sense: the initial layers will be updated so that the
# total loss drops. I don't know if there's gradient signal that tells the
# initial layers to adapt its outputs to match with the inputs of the final
# layers.
#
# What experiment would I run to test this hypothesis?

import copy

import torch

update_all = True

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4, 10, 10, 2

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)

# take snapshot of model state
state0 = copy.deepcopy(model.state_dict())

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4

# only update parameters in first linear layer
if update_all:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    parameters = [p for name, p in model.named_parameters() if name.startswith('0')]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # assert the first layer is updating, while second is frozen
    assert (model.state_dict()['0.bias'] - state0['0.bias']).numpy().any() != 0
    assert (model.state_dict()['0.weight'] - state0['0.weight']).numpy().any() != 0
    if not update_all:
        assert (model.state_dict()['2.bias'] - state0['2.bias']).numpy().all() == 0
        assert (model.state_dict()['2.weight'] - state0['2.weight']).numpy().all() == 0
