import os
import glob
import time
from datetime import datetime

'''Have to import the env folder, this line seems not used, but we actually used it.'''
import CIMProjectEnv
import torch
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
from UncertaintyPPO import PPO
import multiprocessing as mp

train_combination_lists = ['seed50_vac0.9_diss0.9_drl_vs_RxS',
                           'seed50_vac0.9_diss0.9_drl_vs_blocking',
                           'seed50_vac0.9_diss0.9_drl_vs_sub',
                           'seed50_vac0.9_diss0.9_drl_vs_deg',
                           'seed50_vac0.9_diss0.9_drl_vs_drl']

################################### Training ###################################
def train():
    print("============================================================================================")
    train_combination = train_combination_lists[4]

    ####### initialize environment hyperparameters ######
    # env_name = 'CartPole-v1'
    env_name = "InfluenceSpreadEnv-v2"


    max_ep_len = 50  # max timesteps in one episode, larger than number of seeds
    max_training_timesteps = 500 * max_ep_len  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 2  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(500)  # save model frequency (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 1  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)
    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/' + train_combination + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    '''log the details at each step, for debug purpose only'''
    #####################################################
    log_dir_step = "PPO_logs_step"
    if not os.path.exists(log_dir_step):
        os.makedirs(log_dir_step)

    log_dir_step = log_dir_step + '/' + env_name + '/' + train_combination + '/'
    if not os.path.exists(log_dir_step):
        os.makedirs(log_dir_step)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir_step))[2]
    run_num_step = len(current_num_files)

    #### create new log file for each run
    log_f_name_step = log_dir_step + '/PPO_' + env_name + "_log_" + str(run_num_step) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num_step)
    print("logging at : " + log_f_name_step)
    #####################################################
    '''Ends logging each step'''

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory_def = "PPO_preTrained/" + env_name + '/defender/' + train_combination + '/'
    if not os.path.exists(directory_def):
        os.makedirs(directory_def)

    # current_num_files_checkpoint = next(os.walk(directory_def))[2]
    # run_num_pretrained = len(current_num_files_checkpoint)

    checkpoint_path_def = directory_def + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path for defender : " + checkpoint_path_def)

    directory_att = "PPO_preTrained/" + env_name + '/attacker/' + train_combination + '/'
    if not os.path.exists(directory_att):
        os.makedirs(directory_att)

    # # run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder
    # current_num_files_checkpoint = next(os.walk(directory_att))[2]
    # run_num_pretrained = len(current_num_files_checkpoint)

    checkpoint_path_att = directory_att + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path for defender : " + checkpoint_path_att)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", run_num)
        torch.manual_seed(run_num)
        # env.seed(random_seed)
        np.random.seed(run_num)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent_def = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    ppo_agent_att = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,def reward,att reward,def num node,att num node,u num node,'
                'avg b,avg d,avg u,def uncertainty,att uncertainty,def dissonance,att dissonance,'
                'def entropy,att entropy,def explore times,att explore times\n')


    '''This is for logging step'''
    log_f_step = open(log_f_name_step, "w+")
    log_f_step.write('episode,timestep,def reward,att reward,state,'
                     'def action,def action dist,def UM opinion,att action,att action dist,att UM opinion,'
                     'def uncertainty,att uncertainty,def dissonance,att dissonance,'
                     'def entropy,att entropy,def explore,att explore,def picked node,att picked node\n')
    '''Ending logging step'''

    # printing variables
    print_running_reward_def = 0
    print_running_reward_att = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward_def = 0
        current_ep_reward_att = 0
        current_ep_uncertainty_def = 0
        current_ep_uncertainty_att = 0
        current_ep_dissonance_def = 0
        current_ep_dissonance_att = 0
        current_ep_entropy_def = 0
        current_ep_entropy_att = 0
        current_ep_explore_times_def = 0
        current_ep_explore_times_att = 0
        current_ep_steps = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action_att, action_dist_att, opinion_after_UM_att, uncertainty_att, dissonance_att, entropy_att, explore_att = ppo_agent_att.select_action(state)
            action_def, action_dist_def, opinion_after_UM_def, uncertainty_def, dissonance_def, entropy_def, explore_def = ppo_agent_def.select_action(state)
            state, reward, done, info = env.step_two_agents(action_def, action_att)

            reward_def = reward[0]
            reward_att = reward[1]

            log_f_step.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i_episode, time_step, reward_def, reward_att, state,
                                                                         action_def, action_dist_def, opinion_after_UM_def[0:4],
                                                                         action_att, action_dist_att, opinion_after_UM_att[0:4],
                                                                         uncertainty_def, uncertainty_att,dissonance_def, dissonance_att,
                                                                         entropy_def, entropy_att, explore_def, explore_att,
                                                                         info['picked_T_node'], info['picked_F_node']))

            # saving reward and is_terminals
            ppo_agent_att.buffer.rewards.append(reward_att)
            ppo_agent_att.buffer.is_terminals.append(done)
            ppo_agent_def.buffer.rewards.append(reward_def)
            ppo_agent_def.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_steps += 1
            current_ep_reward_def += reward_def
            current_ep_reward_att += reward_att
            current_ep_uncertainty_def += uncertainty_def
            current_ep_uncertainty_att += uncertainty_att
            current_ep_dissonance_def += dissonance_def
            current_ep_dissonance_att += dissonance_att
            current_ep_entropy_def += entropy_def
            current_ep_entropy_att += entropy_att
            current_ep_explore_times_def += explore_def
            current_ep_explore_times_att += explore_att

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent_att.update()
                ppo_agent_def.update()

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward_def = print_running_reward_def / print_running_episodes
                print_avg_reward_att = print_running_reward_att / print_running_episodes
                print_avg_reward_def = round(print_avg_reward_def, 2)
                print_avg_reward_att = round(print_avg_reward_att, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward for defender : {} and attacker : {}".format(i_episode, time_step,
                                                                                print_avg_reward_def, print_avg_reward_att))

                print_running_reward_def = 0
                print_running_reward_att = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model")
                ppo_agent_att.save(checkpoint_path_att)
                ppo_agent_def.save(checkpoint_path_def)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                log_avg_reward_def = current_ep_reward_def / current_ep_steps
                log_avg_reward_att = current_ep_reward_att / current_ep_steps
                log_avg_reward_def = round(log_avg_reward_def, 4)
                log_avg_reward_att = round(log_avg_reward_att, 4)
                log_avg_uncertainty_def = current_ep_uncertainty_def / current_ep_steps
                log_avg_uncertainty_att = current_ep_uncertainty_att / current_ep_steps
                log_avg_uncertainty_def = round(log_avg_uncertainty_def, 4)
                log_avg_uncertainty_att = round(log_avg_uncertainty_att, 4)
                log_avg_dissonance_def = current_ep_dissonance_def / current_ep_steps
                log_avg_dissonance_att = current_ep_dissonance_att / current_ep_steps
                log_avg_dissonance_def = round(log_avg_dissonance_def, 4)
                log_avg_dissonance_att = round(log_avg_dissonance_att, 4)
                log_avg_entropy_def = current_ep_entropy_def / current_ep_steps
                log_avg_entropy_att = current_ep_entropy_att / current_ep_steps
                log_avg_entropy_def = round(log_avg_entropy_def, 4)
                log_avg_entropy_att = round(log_avg_entropy_att, 4)
                log_avg_explore_times_def = current_ep_explore_times_def / current_ep_steps
                log_avg_explore_times_att = current_ep_explore_times_att / current_ep_steps
                log_avg_explore_times_def = round(log_avg_explore_times_def, 4)
                log_avg_explore_times_att = round(log_avg_explore_times_att, 4)

                # This is for CIM environment
                log_f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward_def, log_avg_reward_att,
                                                                           info['num_T_node'], info['num_F_node'], info['num_U_node'],
                                                                           info['avg_b'], info['avg_d'], info['avg_u'],
                                                                           log_avg_uncertainty_def, log_avg_uncertainty_att,
                                                                           log_avg_dissonance_def, log_avg_dissonance_att,
                                                                           log_avg_entropy_def, log_avg_entropy_att,
                                                                           log_avg_explore_times_def, log_avg_explore_times_att))
                log_f.flush()

                current_ep_steps = 0
                break
        print_running_reward_def += current_ep_reward_def
        print_running_reward_att += current_ep_reward_att
        print_running_episodes += 1

        i_episode += 1
        # print("episode is: ", i_episode)

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    for i in range(0, 1):
        train()
    # writer = Sum