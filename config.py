L = 7
H = 6
CPUCT = 4
tau = 1
tau_zero_self_play = 18
CPUS = 20
max_iterations = 1000

# ----------------------------------------------------------------------#
# MCTS parameters
SIM_NUMBER = 30

favorlonggames = True
long_game_factor = 0.1
use_nn_for_terminal = False
use_counter_in_mcts_nn = False
maskinmcts = False

# ----------------------------------------------------------------------#


net = 'resnet'
res_tower = 1
convsize = 256
polfilters = 2
valfilters = 1
usehiddenpol = False
hiddensize = 256

# ----------------------------------------------------------------------#
optim = 'sgd'
sgd_lr = 0.001

# adam_lr=0.01

lrdecay = 2
annealing = 30

use_cuda = True
momentum = 0.9
wdecay = 0.0001  # weight decay
EPOCHS = 10
MINIBATCH = 32
MAXMEMORY = MINIBATCH * 6000
MAXBATCHNUMBER = 10000
MINBATCHNUMBER = 64

# ----------------------------------------------------------------------#
# self play options
dirichlet_for_self_play = True
alpha_dir = 1
epsilon_dir = 0.75
selfplaygames = 400  # i'd recommend at least 64

# see main functions :
use_z_last = False
data_extension = True

# temperatures
tau_self_play = 1  # temperature

# ----------------------------------------------------------------------#
tournamentloop = 2
threshold = 0.51
sim_number_tournaments = 49
tau_zero_eval_new_nn = 1
tau_pv = 1
alternplayer = True

useprevdata = False

# ----------------------------------------------------------------------#
use_counter_in_pure_mcts = False
printstatefreq = 1
checkpoint_frequency = 1


# ----------------------------------------------------------------------#

def particular_states():
    list = [[[], []], [[24], []], [[24], [25]],
            [[24, 26], [25]], [[24, 26], [25, 27]], [[24, 26, 28], [25, 27]], [[24, 26, 28], [25, 27, 29]]]
    return list


def getstate(i):
    list = particular_states()
    elem = list[i]
    yellowboard = 0
    redboard = 0
    yellow = elem[0]
    red = elem[1]
    for u in yellow:
        yellowboard += 2 ** u
    for v in red:
        redboard += 2 ** v
    playerturn = 1
    if len(yellow) != len(red):
        playerturn = - playerturn

    return [yellowboard, redboard, playerturn]
