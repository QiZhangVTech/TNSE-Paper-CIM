import random
import gym
from gym import spaces
import numpy as np
from HelpFunctions import MyNetwork
import networkx as nx

# from .Action import Actions

N_DISCRETE_ACTIONS = 4

# c dataset
# G = nx.read_edgelist("Datasets/email_processed.txt")
# G = nx.read_edgelist("Datasets/facebook_combined.txt")
G = nx.read_edgelist("Datasets/facebook_page.txt")
# G = nx.read_edgelist("Datasets/twitter_combined.txt")
# G = nx.read_edgelist("Datasets/test graph 10.txt")

# Graph = MyNetwork(G)
# Graph.adding_attributes()

class InfluenceSpreadEnv_v2(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.Graph = MyNetwork(G)
        self.Graph.adding_attributes()
        # self.visible_percentage = 0.7
        # self.Graph.generate_partially_observable(self.visible_percentage)

        self.num_T_node = 0
        self.num_F_node = 0
        self.num_U_node = self.Graph.network.number_of_nodes()
        self.previous_T = 0
        self.previous_F = 0
        self.previous_b = self.Graph.get_avg_opinions()[0]
        self.previous_d = self.Graph.get_avg_opinions()[1]

        self.num_free_nodes = self.Graph.get_num_of_free_nodes()
        self.sum_of_edges = self.Graph.get_sum_of_edges_of_free_nodes()
        self.max_degree = self.Graph.get_max_degree_of_free_nodes()
        self.sum_of_b = self.Graph.get_opinions()[2][0]
        self.sum_of_d = self.Graph.get_opinions()[2][1]
        self.max_neighbor_sum_b = self.Graph.get_max_neighbor_b_sum_of_free_nodes()
        self.max_neighbor_sum_d = self.Graph.get_max_neighbor_d_sum_of_free_nodes()
        self.max_RxS = self.Graph.get_max_RxS_of_free_nodes()

        self.num_free_nodes_max = self.num_free_nodes
        self.sum_of_edges_max = self.sum_of_edges
        self.max_degree_max = self.max_degree
        self.sum_of_b_max = self.Graph.node_density
        self.sum_of_d_max = self.Graph.node_density
        self.max_neighbor_sum_b_max = self.max_degree * 1
        self.max_neighbor_sum_d_max = self.max_degree * 1
        self.max_RxS_max = 1

        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4])
        # self.observation_space = gym.spaces.MultiDiscrete([4, 4])  # only keep the second and the third as the states since all others don't change.

        self.count = 0
        self.update_times_att = 1  # when a seed node is picked, how many times it spreads news
        self.update_times_def = 1
        self.random_update_times = 0
        self.uncertainty_maximization_threshold = [0.001, 0.6]
        self.number_of_seed = 50  # for each party

        # This definition is not correct, because action_set is an array of [int], it won't update the four values
        # self.action_set = [self.Graph.unf_node, self.Graph.rnf_node, self.Graph.snf_node, self.Graph.cf_node]

    @property
    def action_set(self):
        return [self.Graph.unf_node, self.Graph.rnf_node, self.Graph.snf_node, self.Graph.cf_node]

    def step_two_agents(self, action_def, action_att):
        picked_node_att, att_UM_times, stop_att = self.take_action_att(action_att)
        picked_node_def, def_UM_times, stop_def = self.take_action_def(action_def)
        self.count += 1
        done = stop_def or stop_att

        for i in range(0, self.random_update_times):
            # print("Random update")
            self.Graph.random_sharing()

        numbers_in_group = self.Graph.count_numbers()
        self.num_T_node = numbers_in_group[1]
        self.num_F_node = numbers_in_group[0]
        self.num_U_node = numbers_in_group[2]

        avg_b, avg_d, avg_u = self.Graph.get_avg_opinions()
        # print(f"Average opinion: ", avg_b, avg_d, avg_u)

        opinions_in_network_T, opinions_in_network_F, opinions_in_network_U = self.Graph.get_opinions()
        # self.total_b[0] = opinions_in_network_T[0]
        # self.total_d[0] = opinions_in_network_T[1]
        # self.total_u[0] = opinions_in_network_T[2]
        #
        # self.total_b[1] = opinions_in_network_F[0]
        # self.total_d[1] = opinions_in_network_F[1]
        # self.total_u[1] = opinions_in_network_F[2]
        #
        # self.total_b[2] = opinions_in_network_U[0]
        # self.total_d[2] = opinions_in_network_U[1]
        # self.total_u[2] = opinions_in_network_U[2]

        self.num_free_nodes = self.Graph.get_num_of_free_nodes()
        self.sum_of_edges = self.Graph.get_sum_of_edges_of_free_nodes()
        self.max_degree = self.Graph.get_max_degree_of_free_nodes()
        self.sum_of_b = self.Graph.get_opinions()[2][0]
        self.sum_of_d = self.Graph.get_opinions()[2][1]
        self.max_neighbor_sum_b = self.Graph.get_max_neighbor_b_sum_of_free_nodes()
        self.max_neighbor_sum_d = self.Graph.get_max_neighbor_d_sum_of_free_nodes()
        self.max_RxS = self.Graph.get_max_RxS_of_free_nodes()

        state = np.array([self.Graph.number_to_level(self.num_free_nodes, self.num_free_nodes_max),
                          self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
                          self.Graph.number_to_level(self.max_degree, self.max_degree_max),
                          self.Graph.number_to_level(self.sum_of_b, self.sum_of_b_max),
                          self.Graph.number_to_level(self.sum_of_d, self.sum_of_d_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_b, self.max_neighbor_sum_b_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_d, self.max_neighbor_sum_d_max),
                          self.Graph.number_to_level(self.max_RxS, self.max_RxS_max)])

        # state = np.array([self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
        #                   self.Graph.number_to_level(self.max_degree, self.max_degree_max)])

        state_numbers = np.array([self.num_T_node, self.num_F_node, self.num_U_node])
        # print(f"current num of T node: {self.num_T_node}, previous num of T node: {self.previous_T}")
        # reward idea 1
        # reward_T = self.num_T_node - self.previous_T
        # reward_F = self.num_F_node - self.previous_F
        # self.previous_T = self.num_T_node
        # self.previous_F = self.num_F_node
        # reward idea 2
        # reward_T = avg_b * 10
        # reward_F = avg_d * 10
        # reward idea 3
        # b_increase = avg_b - self.previous_b
        # d_increase = avg_d - self.previous_d
        # self.previous_b = avg_b
        # self.previous_d = avg_d
        # reward_T = 100 * (b_increase - d_increase)
        # reward_F = 100 * (d_increase - b_increase)
        
        # reward idea 4
        sum_b_T = opinions_in_network_T[0]
        sum_d_F = opinions_in_network_F[1]
        reward_T = sum_b_T
        reward_F = sum_d_F

        # print(f"The reward: \n"
        #       f"b increase is {b_increase}, d increase is {d_increase} \n"
        #       f"average b is {avg_b}, previous b is {self.previous_b} \n"
        #       f"average d is {avg_d}, previous b is {self.previous_d} \n"
        #       f"Reward T party is {reward_T}, F party is {reward_F}")
        # self.Graph.plot_all_opinions_prep()
        # total_b_increase = self.total_b-self.previous_total_b
        # total_d_increase = self.total_d-self.previous_total_d
        # total_u_decrease = self.previous_total_u - self.total_u

        # self.Graph.check_bdu_sum()
        if self.count == self.number_of_seed:
            done = True

        info = {
            'num_T_node': self.num_T_node,
            'num_F_node': self.num_F_node,
            'num_U_node': self.num_U_node,
            'avg_b': avg_b,
            'avg_d': avg_d,
            'avg_u': avg_u,
            'picked_T_node': picked_node_def,
            'picked_F_node': picked_node_att,
            'def_UM_times': def_UM_times,
            'att_UM_times': att_UM_times
        }

        if done:
            #     self.previous_T = 0
            #     self.previous_F = 0
            self.count = 0
            # self.Graph.plot_all_users_opinions()
        #     Graph.adding_attributes()

        return state, [reward_T, reward_F], done, info

    def take_action_def(self, action):
        stop = self.Graph.pick_unf_node()
        self.Graph.pick_rnf_node()
        self.Graph.pick_snf_node()
        self.Graph.pick_cf_node()
        # self.Graph.pick_random_node()

        # DRL agent
        if action == 1:
            # because blockings are different for T and F parties. So we the same action requires two input
            picked_node = self.action_set[action][1]
        else:
            picked_node = self.action_set[action]

        # random strategy
        # picked_node = self.Graph.random_node
        # centrality
        # picked_node = self.Graph.cf_node
        # page rank
        # picked_node = self.Graph.pr_node
        # rnf node
        # picked_node = self.Graph.rnf_node[1]
        # snf node
        # picked_node = self.Graph.snf_node
        # unf node
        # picked_node = self.Graph.unf_node

        self.Graph.pick_TIP(picked_node)
        # print(f"True party action is: {action}, seed node is: {picked_node}")
        for i in range(0, self.update_times_def):
            # print("T party update from seed node")
            UM_times = self.Graph.update_after_pick_seed(picked_node, self.uncertainty_maximization_threshold)
        # avg_b, avg_d, avg_u = self.Graph.get_avg_opinions()
        # print(f"Average opinion: ", avg_b, avg_d, avg_u)

        return picked_node, UM_times, stop

    def take_action_att(self, action):
        stop = self.Graph.pick_unf_node()
        self.Graph.pick_rnf_node()
        self.Graph.pick_snf_node()
        self.Graph.pick_cf_node()
        # self.Graph.pick_random_node()

        # DRL agent
        # if action == 1:
        #     picked_node = self.action_set[action][0]
        # else:
        #     picked_node = self.action_set[action]

        # random strategy
        # picked_node = self.Graph.random_node
        # centrality; deg
        # picked_node = self.Graph.cf_node
        # page rank
        # picked_node = self.Graph.pr_node
        # rnf node; blocking
        # picked_node = self.Graph.rnf_node[0]
        # snf node; subgreedy
        picked_node = self.Graph.snf_node
        # unf node; RxS, active
        # picked_node = self.Graph.unf_node

        self.Graph.pick_FIP(picked_node)
        # print(f"False party action is: {action}, seed node is: {picked_node}")

        for i in range(0, self.update_times_att):
            # print("F party update from seed node")
            UM_times = self.Graph.update_after_pick_seed(picked_node, self.uncertainty_maximization_threshold)
        # avg_b, avg_d, avg_u = self.Graph.get_avg_opinions()
        # print(f"Average opinion: ", avg_b, avg_d, avg_u)

        return picked_node, UM_times, stop

    # here we cannot use step because the step() in gym has only one input action
    # in previous design, the step function is for both parties, so feed two actions.
    def step_def(self, action_def):

        picked_node_def, def_UM_times, stop = self.take_action_def(action_def)
        self.count += 1

        done = stop

        self.num_free_nodes = self.Graph.get_num_of_free_nodes()
        self.sum_of_edges = self.Graph.get_sum_of_edges_of_free_nodes()
        self.max_degree = self.Graph.get_max_degree_of_free_nodes()
        self.sum_of_b = self.Graph.get_opinions()[2][0]
        self.sum_of_d = self.Graph.get_opinions()[2][1]
        self.max_neighbor_sum_b = self.Graph.get_max_neighbor_b_sum_of_free_nodes()
        self.max_neighbor_sum_d = self.Graph.get_max_neighbor_d_sum_of_free_nodes()
        self.max_RxS = self.Graph.get_max_RxS_of_free_nodes()

        state = np.array([self.Graph.number_to_level(self.num_free_nodes, self.num_free_nodes_max),
                          self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
                          self.Graph.number_to_level(self.max_degree, self.max_degree_max),
                          self.Graph.number_to_level(self.sum_of_b, self.sum_of_b_max),
                          self.Graph.number_to_level(self.sum_of_d, self.sum_of_d_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_b, self.max_neighbor_sum_b_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_d, self.max_neighbor_sum_d_max),
                          self.Graph.number_to_level(self.max_RxS, self.max_RxS_max)])

        # state = np.array([self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
        #                   self.Graph.number_to_level(self.max_degree, self.max_degree_max)])

        # state_numbers = np.array([self.num_T_node, self.num_F_node, self.num_U_node])
        # print(f"current num of T node: {self.num_T_node}, previous num of T node: {self.previous_T}")
        reward_T = self.num_T_node - self.previous_T
        self.previous_T = self.num_T_node
        # self.previous_F = self.num_F_node
        # self.Graph.plot_all_opinions_prep()
        # total_b_increase = self.total_b-self.previous_total_b
        # total_d_increase = self.total_d-self.previous_total_d
        # total_u_decrease = self.previous_total_u - self.total_u

        # self.Graph.check_bdu_sum()
        if self.count == 2 * self.number_of_seed:
            done = True

        info = np.array([self.num_T_node, self.num_F_node, self.num_U_node, picked_node_def, def_UM_times])

        if done:
            #     self.previous_T = 0
            #     self.previous_F = 0
            self.count = 0
            # self.Graph.plot_all_users_opinions()
        #     Graph.adding_attributes()

        return state, reward_T, done, info

    def step_att(self, action_att):

        picked_node_att, att_UM_times, stop = self.take_action_att(action_att)
        self.count += 1

        done = stop
        self.num_free_nodes = self.Graph.get_num_of_free_nodes()
        self.sum_of_edges = self.Graph.get_sum_of_edges_of_free_nodes()
        self.max_degree = self.Graph.get_max_degree_of_free_nodes()
        self.sum_of_b = self.Graph.get_opinions()[2][0]
        self.sum_of_d = self.Graph.get_opinions()[2][1]
        self.max_neighbor_sum_b = self.Graph.get_max_neighbor_b_sum_of_free_nodes()
        self.max_neighbor_sum_d = self.Graph.get_max_neighbor_d_sum_of_free_nodes()
        self.max_RxS = self.Graph.get_max_RxS_of_free_nodes()

        state = np.array([self.Graph.number_to_level(self.num_free_nodes, self.num_free_nodes_max),
                          self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
                          self.Graph.number_to_level(self.max_degree, self.max_degree_max),
                          self.Graph.number_to_level(self.sum_of_b, self.sum_of_b_max),
                          self.Graph.number_to_level(self.sum_of_d, self.sum_of_d_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_b, self.max_neighbor_sum_b_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_d, self.max_neighbor_sum_d_max),
                          self.Graph.number_to_level(self.max_RxS, self.max_RxS_max)])

        # state = np.array([self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
        #                   self.Graph.number_to_level(self.max_degree, self.max_degree_max)])

        state_numbers = np.array([self.num_T_node, self.num_F_node, self.num_U_node])
        # reward_T = self.num_T_node - self.previous_T
        # print(f"current num of F node: {self.num_F_node}, previous num of F node: {self.previous_F}")
        reward_F = self.num_F_node - self.previous_F
        self.previous_F = self.num_F_node
        # self.Graph.plot_all_opinions_prep()
        # total_b_increase = self.total_b-self.previous_total_b
        # total_d_increase = self.total_d-self.previous_total_d
        # total_u_decrease = self.previous_total_u - self.total_u

        # self.Graph.check_bdu_sum()
        if self.count == 2 * self.number_of_seed:
            done = True

        info = np.array([self.num_T_node, self.num_F_node, self.num_U_node, picked_node_att, att_UM_times])

        if done:
            #     self.previous_T = 0
            #     self.previous_F = 0
            self.count = 0
            # self.Graph.plot_all_users_opinions()
        #     Graph.adding_attributes()

        return state, reward_F, done, info

    def reset(self):
        self.num_T_node = 0
        self.num_F_node = 0
        self.num_U_node = self.Graph.node_density
        self.count = 0
        self.previous_T = 0
        self.previous_F = 0

        self.Graph.adding_attributes()
        # self.Graph.generate_partially_observable(self.visible_percentage)
        
        self.previous_b = self.Graph.get_avg_opinions()[0]
        self.previous_d = self.Graph.get_avg_opinions()[1]

        self.num_free_nodes = self.Graph.get_num_of_free_nodes()
        self.sum_of_edges = self.Graph.get_sum_of_edges_of_free_nodes()
        self.max_degree = self.Graph.get_max_degree_of_free_nodes()
        self.sum_of_b = self.Graph.get_opinions()[2][0]
        self.sum_of_d = self.Graph.get_opinions()[2][1]
        self.max_neighbor_sum_b = self.Graph.get_max_neighbor_b_sum_of_free_nodes()
        self.max_neighbor_sum_d = self.Graph.get_max_neighbor_d_sum_of_free_nodes()
        self.max_RxS = self.Graph.get_max_RxS_of_free_nodes()

        state = np.array([self.Graph.number_to_level(self.num_free_nodes, self.num_free_nodes_max),
                          self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
                          self.Graph.number_to_level(self.max_degree, self.max_degree_max),
                          self.Graph.number_to_level(self.sum_of_b, self.sum_of_b_max),
                          self.Graph.number_to_level(self.sum_of_d, self.sum_of_d_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_b, self.max_neighbor_sum_b_max),
                          self.Graph.number_to_level(self.max_neighbor_sum_d, self.max_neighbor_sum_d_max),
                          self.Graph.number_to_level(self.max_RxS, self.max_RxS_max)])

        # state = np.array([self.Graph.number_to_level(self.sum_of_edges, self.sum_of_edges_max),
        #                   self.Graph.number_to_level(self.max_degree, self.max_degree_max)])

        return state

# from stable_baselines3.common.env_checker import check_env
# env = InfluenceSpreadEnv_v2()
# check_env(env)
# print(f"Action Space: {env.action_space}")
# print(f"Action Space Sample: {env.action_space.sample()}")
