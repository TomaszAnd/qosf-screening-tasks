import pennylane as qml
from pennylane import numpy as np
from tqdm.notebook import tqdm


from scipy.stats import unitary_group
import numpy.random as rnd

# Fix seeds
seeds = [7658741,7658742,7658743,7658744]

dev1 = qml.device("default.qubit", wires=4)

def random_state(N, seed):
    """Create a random state on N qubits."""
    states = unitary_group.rvs(2 ** N, random_state=rnd.default_rng(seed))
    return states[0]

# Create a list of 4 4-qubit random initial states
random_states_list = []
for seed in seeds:
    random_states_list.append(random_state(4, seed))

# Create a list of 4 4-qubit target states
indices = [3, 5, 10, -4]
target_states_list = []
for index in indices:
    target_state = np.zeros([2 ** 4])
    target_state[index] = 1
    target_states_list.append(target_state)

@qml.qnode(dev1)
def circuit(params, state_index):

    random_states_list[state_index]

    for j in range(2): # 2 layers
        for i in range(4): # 4 qubits
            qml.Rot(*params[j][i], wires=i)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[2,3])
        qml.CNOT(wires=[1,2])

    density = np.outer(target_states_list[state_index], target_states_list[state_index])
    return qml.expval(qml.Hermitian(density, wires=[0,1,2,3]))


def cost(var,state_index):
    return 1-circuit(var,state_index)

#initialize parameters
init_params = np.random.rand(2, 4, 3) # 2 layers, 4 qubits, 3 parameters per rotation
state_index = 0
print(cost(init_params,state_index))

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4) # stepsize is the learning rate

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in tqdm(range(steps)):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 10 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params,state_index)))

print("Optimized rotation angles: {}".format(params))

circuit(params,state_index)
dev1.state