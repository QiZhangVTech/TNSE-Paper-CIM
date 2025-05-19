import random
import gym
from gym import spaces
import numpy as np
from HelpFunctions import MyNetwork
import networkx as nx

# from .Action import Actions

N_DISCRETE_ACTIONS = 4

# c dataset
G = nx.read_edgelist("Datasets/email_processed.txt")
# G = nx.read_edgelist("Datasets/facebook_combined.txt")
# G = nx.read_edgelist("Datasets/facebook_page.txt")
# G = nx.read_edgelist("Datasets/twitter_combined.txt")
# G = nx.read_edgelist("Datasets/test graph 10.txt")

# Graph = MyNetwork(G)
# Graph.adding_attributes()

class InfluenceSpreadEnv_v1(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.Graph = MyNetwork(G)
        self.Graph.adding_attributes()

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
        self.update_times_def = 1  # when a seed node is picked, how many times it spreads news
        self.random_update_times = 0
        self.uncertainty_maximization_threshold = [0.001, 0.6]
        self.number_of_seed = 40  # for each party

        # This definition is not correct, because action_set is an array of [int], it won't update the four values
        # self.action_set = [self.Graph.unf_node, self.Graph.rnf_node, self.Graph.snf_node, self.Graph.cf_node]

    @property
    def action_set(self):
        return [self.Graph.unf_node, self.Graph.rnf_node, self.Graph.snf_node, self.Graph.cf_node]

    def take_action(self, action):
        stop = self.Graph.pick_unf_node()
        self.Graph.pick_rnf_node()
        self.Graph.pick_snf_node()
        self.Graph.pick_cf_node()
        # self.Graph.pick_random_node()
        # print("node for each type: ")
        # print("self.Graph.unf_node: ", self.Graph.unf_node)
        # print("self.Graph.rnf_node: ", self.Graph.rnf_node)
        # print("self.Graph.snf_node: ", self.Graph.snf_node)
        # print("self.Graph.cf_node: ", self.Graph.cf_node)

        # print("action is: ", action)
        # DRL agent
        if action == 1:
            # because blockings are different for T and F parties. So we the same action requires two input
            picked_node = self.action_set[action][1]
        else:
            picked_node = self.action_set[action]

        # print("action is: ", action)
        # print("picked node is: ", picked_node)

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


    # here we cannot use step because the step() in gym has only one input action
    # in previous design, the step function is for both parties, so feed two actions.
    def step(self, action):

        picked_node, UM_times, stop = self.take_action(action)
        self.count += 1

        done = stop
        # done = False

        numbers_in_group = self.Graph.count_numbers()
        self.num_T_node = numbers_in_group[1]
        self.num_F_node = numbers_in_group[0]
        self.num_U_node = numbers_in_group[2]

        avg_b, avg_d, avg_u = self.Graph.get_avg_opinions()

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
        # reward = self.num_T_node - self.previous_T
        # self.previous_T = self.num_T_node

        # reward = avg_b-self.previous_b

        reward = avg_b * 10

        # self.previous_F = self.num_F_node
        # self.Graph.plot_all_opinions_prep()
        # total_b_increase = self.total_b-self.previous_total_b
        # total_d_increase = self.total_d-self.previous_total_d
        # total_u_decrease = self.previous_total_u - self.total_u

        # self.Graph.check_bdu_sum()
        if self.count == self.number_of_seed:
            done = True

        # info = np.array([self.num_T_node, self.num_U_node, picked_node, UM_times])

        info = {
            'num_T_node': self.num_T_node,
            'num_U_node': self.num_U_node,
            'avg_b': avg_b,
            'avg_d': avg_d,
            'avg_u': avg_u,
            'picked_node': picked_node,
            'UM_times': UM_times
        }

        if done:
            #     self.previous_T = 0
            #     self.previous_F = 0
            self.count = 0
            # self.Graph.plot_all_users_opinions()
        #     Graph.adding_attributes()

        return state, reward, done, info

    def reset(self):
        self.num_T_node = 0
        self.num_F_node = 0
        self.num_U_node = self.Graph.node_density
        self.count = 0
        self.previous_T = 0
        self.previous_F = 0

        self.Graph.adding_attributes()
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
# env = InfluenceSpreadEnv_v1()
# check_env(env)
# print(f"Action Space: {env.action_space}")
# print(f"Action Space Sample: {env.action_space.sample()}")
