#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             Main.py
# Description:      Main program
# Authors:          Jean-Philippe Bruneton & Adle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import time
import numpy as np
import config
import main_functions
import random
import copy
import torch.utils
from torchsummary import summary


# =================================== MAIN ==================================== #

def launch():
    # --------------------------------------------------------------------- #
    # init data generated by self play
    dataseen = np.zeros((3 * config.L * config.H + config.L + 1))

    # --------------------------------------------------------------------- #
    # Init Neural Net
    best_player_so_far = main_functions.load_or_create_neural_net()

    # --------------------------------------------------------------------- #
    # init ELO ratings and improvement counters
    elos = [0]
    improved = 0
    total_improved = 0

    # --------------------------------------------------------------------- #
    # print summary. For some reasons I don't understand it may produce bugs depending on the machine used.
    # see end of config file to see the summary for default parameters

    printsummary = 0
    if printsummary:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config.net == 'densenet':
            print(summary(best_player_so_far.to(device), (1, 3 * config.H * config.L)))
        if config.net == 'resnet':
            print(summary(best_player_so_far.to(device), (3, config.H, config.L)))
        time.sleep(0.1)

    # --------------------------------------------------------------------- #
    # This displays the probabilities and NN reward for the first moves as described
    # in the readme file

    showinit = 1
    if showinit:
        print('init is')
        _ = main_functions.printstates(best_player_so_far)
        elos = main_functions.geteloratings(elos, best_player_so_far, 1, total_improved)

    # --------------------------------------------------------------------- #
    # Main loop

    i = 0

    while i < config.max_iterations:

        random.seed()
        np.random.seed()

        # Number of simulations for self play increases as the NN gets better, until = 350 ( ~ 7^3). I think it is a good idea (?) though it does slow down the entire process.
        sim_number = config.SIM_NUMBER + i
        if sim_number > 350:
            sim_number = 350

        # generate data from self play
        use_this_data, prev_data_seen = main_functions.generate_self_play_data(best_player_so_far, sim_number, dataseen,
                                                                               i)

        # deepcopy last best_model
        previous_best = copy.deepcopy(best_player_so_far)

        # neural net training by experience replay :
        print('')
        print('--- Improving model ---')

        main_functions.improve_model_resnet(best_player_so_far, use_this_data, total_improved)
        best_player_so_far.eval()

        # Check wether the model has improved
        print('')
        print('--- Winrate against previous model ---')
        time.sleep(0.01)

        use_dirichlet = False
        winp1, winp2, draws, ratio = main_functions.play_v1_against_v2 \
            (best_player_so_far, previous_best, config.tournamentloop, config.CPUS, config.sim_number_tournaments,
             config.CPUCT,
             config.tau_pv, config.tau_zero_eval_new_nn, use_dirichlet)
        time.sleep(0.01)
        # print('FYI, first player won by', int(1000*ratio)/10, '%' )

        # Model has improved, then save model, save data:
        if (winp1 + draws / 2) / (winp1 + winp2 + draws) >= config.threshold:
            improved = 1
        else:
            improved = 0
        if improved == 1:
            total_improved += 1
            time.sleep(0.01)
            print('model has improved with score', 100 * (winp1 + draws / 2) / (winp1 + winp2 + draws),
                  '% in', config.CPUS * config.tournamentloop, 'games')
            print('')

            # it becomes the new best model
            if config.net == 'densenet':
                torch.save(best_player_so_far.state_dict(), './best_model_densenet.pth')
            if config.net == 'resnet':
                torch.save(best_player_so_far.state_dict(), './best_model_resnet.pth')

            # this data was good data since it improved the NN : so we stack it to the previous data of self play
            if config.useprevdata:
                if use_this_data.shape[0] > config.MAXMEMORY:
                    dataseen = np.copy(use_this_data[-config.MAXMEMORY:-1, :])
                else:
                    dataseen = np.copy(use_this_data)
            # print the new NN values for particular states
            if i % config.printstatefreq == config.printstatefreq - 1:
                getbreak = main_functions.printstates(best_player_so_far)


        # Model has not improved : throw everything away and start again:
        else:
            time.sleep(0.01)
            print('model has not improved enough, score is only', 100 * (winp1 + draws / 2) / (winp1 + winp2 + draws),
                  '%')
            best_player_so_far = previous_best
            best_player_so_far.eval()

            # in particular, dont save this data
            if config.useprevdata:
                dataseen = np.copy(prev_data_seen)
            getbreak = 0

        # finally, print statistics about the learning, and ELO ratings, we make games against NN and pure MCTS
        elos = main_functions.geteloratings(elos, best_player_so_far, improved, total_improved)

        # that makes the program quit when the model is good enough
        if getbreak == 1 and elos[-1] > 1800:
            i = i + config.max_iterations
            # thus breaking loop

        # ---------------------------------------------------------------------#
        # end main loop
        i += 1
        time.sleep(0.001)


if __name__ == '__main__':
    launch()
