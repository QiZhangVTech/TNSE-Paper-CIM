import pickle
import random
import time
import math

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import os

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import sys
# sys.stdout = open('output_1.txt', 'w')

class MyNetwork:

    def __init__(self, G):
        self.network = G
        self.node_density = G.number_of_nodes()
        self.num_of_edges = G.number_of_edges()

        self.average_bs = []
        self.average_ds = []
        self.average_us = []
        self.plot_average_opinion_times = []

        self.arr_b = []
        self.arr_d = []
        self.arr_u = []
        self.arr_b_T = []
        self.arr_d_T = []
        self.arr_u_T = []
        self.arr_b_F = []
        self.arr_d_F = []
        self.arr_u_F = []
        self.arr_b_U = []
        self.arr_d_U = []
        self.arr_u_U = []

        self.plot_all_opinions_times = []

        self.unf_node = random.choice(list(G.nodes))
        self.rnf_node = [random.choice(list(G.nodes)), random.choice(list(G.nodes))]
        self.snf_node = random.choice(list(G.nodes))
        self.cf_node = random.choice(list(G.nodes))
        self.pr_node = random.choice(list(G.nodes))
        self.random_node = random.choice(list(G.nodes))

        self.True_seeds = []
        self.False_seeds = []

        self.UM_trigger_times = 0  # How many times Uncertainty Maximization is triggered

    def adding_attributes(self):
        # change those mu and sigma to modify the reading and sharing frequency
        mu_always = 1
        mu_high = 0.75
        sigma_novariance = 0
        sigma = 0.25
        behavior_freq = [0.1, 0.5, 0.75, 1.0]

        # prob_UOM = 1 # 1 means all nodes are UOM node, 0 means all nodes are HOM node. 0.3 means 30% nodes are UOM
        # target_users = self.pick_highest_centrality_users(percentage=prob_UOM)

        r = 1
        s = 1
        w = 101

        for node in self.network.nodes:

            # randomly assign UOM users
            # node_type = random.choices([0, 1], weights=(100 * prob_UOM, 100 * (1 - prob_UOM)), k=1)[0]

            # targeting the users who has higher centrality to be the UOM users
            # if node in target_users:
            #     node_type = 0  # 0 is UOM
            # else:
            #     node_type = 1  # 1 is HOM
            # print(f"node {node} has type {node_type}")
            node_type = 0  # 0 is UOM
            # node_type = 1  # 1 is HOM
            # no trust based
            # node_type = 2

            # reading_prob = random.gauss(mu_high, sigma)
            # if reading_prob > 1:
            #     # print(f"The reading prob {reading_prob} is greater than 1")
            #     reading_prob = 1
            # sharing_prob = random.gauss(mu_high, sigma)
            # if sharing_prob > 1:
            #     # print(f"The sharing prob {sharing_prob} is greater than 1")
            #     sharing_prob = 1
            # reading_prob = np.random.choice(behavior_freq)
            # sharing_prob = np.random.choice(behavior_freq)
            # filtering_capability = random.gauss(mu=0.2, sigma=0.2)
            # filtering_capability = 0

            reading_prob = np.random.uniform(low=0.7, high=1.0, size=None)
            sharing_prob = np.random.uniform(low=0.7, high=1.0, size=None)
            filtering_capability = np.random.uniform(low=0.0, high=1.0, size=None)
            # while True:
            #     filtering_capability = random.gauss(mu_high, sigma)
            #     if 0 <= filtering_capability <= 1:
            #         break
            b = r / (r + s + w)
            d = s / (r + s + w)
            u = w / (r + s + w)
            a = 0.5

            # we cannot use P_b and P_d to decide
            # because at the beginning, P_b and P_d are roughly 0.5 and all nodes are in True party
            if b > 0.5:
                role = 1  # True
            elif d > 0.5:
                role = 0  # False
            else:
                role = 2  # Uncertain

            attrs = {node: {"role": role, "color": "white",
                            "b": b, "d": d, "u": u, "a": a,
                            "P_r": reading_prob, "P_s": sharing_prob, "P_f": filtering_capability,
                            "free_node_or_not": 1, # all nodes are free initially
                            "seed_or_not": 0, "expected_role": role,
                            "node_type": node_type}}  # all nodes are not seeds at the beginning
            # role: plain bdu, expected role: expected bdu
            nx.set_node_attributes(self.network, attrs)

        # initialize at time 0
        # self.plot_average_opinions_prep()
        # self.plot_all_opinions_prep()

    def generate_partially_observable(self, visible_percentage):
        remove_or_not = [0, 1]
        for edge in self.network.edges:
            remove_action = \
            random.choices(remove_or_not, weights=(100 * visible_percentage, 100 * (1 - visible_percentage)), k=1)[0]
            if remove_action:
                u = edge[0]
                v = edge[1]
                self.network.remove_edge(u, v)
        print(f"Partially Observable network with {visible_percentage} generated successfully!")

    def pick_highest_centrality_users(self, percentage):
        nodes_degree = nx.degree_centrality(self.network)
        # print(nodes_degree)
        target_users = []
        num_picked_users = percentage * self.node_density
        print("number of picked users: ", num_picked_users)
        while len(nodes_degree) > self.node_density - num_picked_users:
            max_key = max(nodes_degree, key=nodes_degree.get)
            target_users.append(max_key)
            nodes_degree.pop(max_key)

        return target_users

    def educate_users(self, educate_percentage):
        nodes_degree = nx.degree_centrality(self.network)
        # print(nodes_degree)
        target_educate_users = []
        num_educated_users = educate_percentage * self.node_density
        print("number of educated users: ", num_educated_users)
        while len(nodes_degree) > self.node_density - num_educated_users:
            max_key = max(nodes_degree, key=nodes_degree.get)
            target_educate_users.append(max_key)
            nodes_degree.pop(max_key)

        # print(target_educate_users)
        print(len(target_educate_users))
        r = 1
        s = 1
        w = 101
        for node in self.network.nodes:
            d = s / (r + s + w)
            b = np.random.uniform(low=0.5, high=1.0 - d, size=None)
            u = 1 - b - d
            role = 1
            seed_or_not = 1
            if node in target_educate_users:
                # print(node)
                attrs = {node: {"role": role, "color": "white",
                                "b": b, "d": d, "u": u,
                                "seed_or_not": seed_or_not, "expected_role": role}}
                # role: plain bdu, expected role: expected bdu
                nx.set_node_attributes(self.network, attrs)
                self.update_after_pick_seed(start_node=node, uncertainty_maximization_threshold=[0.001, 0.6])

    def get_num_of_free_nodes(self):
        count = 0
        for i in self.network.nodes:
            if self.network.nodes[i].get("free_node_or_not") != 1:
                continue
            else:
                count = count + 1
        return count

    def get_sum_of_edges_of_free_nodes(self):
        sum = 0
        for edge in self.network.edges:
            free_or_not_0 = self.network.nodes[edge[0]].get("free_node_or_not")
            free_or_not_1 = self.network.nodes[edge[1]].get("free_node_or_not")
            if free_or_not_0 == 1 and free_or_not_1 == 1:
                sum = sum + 1
        return sum

    def get_max_degree_of_free_nodes(self):
        max_value = 0
        nodes_degree = nx.degree_centrality(self.network)
        while len(nodes_degree) > 0:
            max_key = max(nodes_degree, key=nodes_degree.get)
            if self.network.nodes[max_key].get("free_node_or_not") == 1:
                max_value = nodes_degree.get(max_key)
                break
            else:
                nodes_degree.pop(max_key)
        return max_value

    def get_number_of_free_neighbors(self, node):
        num_of_free_neighbors = 0
        for i in self.network.neighbors(node):
            if self.network.nodes[i].get("free_node_or_not") == 1:
                num_of_free_neighbors = num_of_free_neighbors + 1
            else:
                continue
        return num_of_free_neighbors

    def get_max_neighbor_b_sum_of_free_nodes(self):
        max_b_sum = 0

        for i in self.network.nodes:
            sum_b_of_neighbors = 0
            # if self.network.nodes[i].getet("seed_or_not") == 1 or self.network.nodes[i].get("role") != 2:
            if self.network.nodes[i].get("free_node_or_not") != 1:
                continue
            else:
                for node in self.network.neighbors(i):
                    sum_b_of_neighbors = sum_b_of_neighbors + self.network.nodes[node].get("b")
                if sum_b_of_neighbors > max_b_sum:
                    max_b_sum = sum_b_of_neighbors

        return max_b_sum

    def get_max_neighbor_d_sum_of_free_nodes(self):
        max_d_sum = 0

        for i in self.network.nodes:
            sum_d_of_neighbors = 0
            # if self.network.nodes[i].getet("seed_or_not") == 1 or self.network.nodes[i].get("role") != 2:
            if self.network.nodes[i].get("free_node_or_not") != 1:
                continue
            else:
                for node in self.network.neighbors(i):
                    sum_d_of_neighbors = sum_d_of_neighbors + self.network.nodes[node].get("d")
                if sum_d_of_neighbors > max_d_sum:
                    max_d_sum = sum_d_of_neighbors

        return max_d_sum

    def get_max_RxS_of_free_nodes(self):
        max_RxS = 0
        for i in self.network.nodes:
            # if self.network.nodes[i].getet("seed_or_not") == 1 or self.network.nodes[i].get("role") != 2:
            if self.network.nodes[i].get("free_node_or_not") != 1:
                continue
            else:
                reading_prob = self.network.nodes[i].get("P_r")
                sharing_prob = self.network.nodes[i].get("P_s")
                RxS = reading_prob * sharing_prob
                if RxS > max_RxS:
                    max_RxS = RxS
        return max_RxS

    def count_numbers(self):
        numbers_in_groups = [0, 0, 0]  # false, true, uncertain
        for node in self.network.nodes(data=True):
            numbers_in_groups[node[1].get("expected_role")] += 1
            # numbers_in_groups[node[1].get("role")] += 1
        # print("numbers in groups: ", numbers_in_groups)
        return numbers_in_groups

    def get_avg_opinions(self):
        b_in_network = 0
        d_in_network = 0
        u_in_network = 0
        for node in self.network.nodes(data=True):
            b_in_network += node[1].get("b")
            d_in_network += node[1].get("d")
            u_in_network += node[1].get("u")
            # print("For node: ", node[0])
            # print("This b is: ", node[1].get("b"))
            # print("This d is: ", node[1].get("d"))
            # print("This u is: ", node[1].get("u"))
            # print("b, d, u are: ", b_in_network, d_in_network, u_in_network)
        avg_b = b_in_network/self.node_density
        avg_d = d_in_network/self.node_density
        avg_u = u_in_network/self.node_density
        return avg_b, avg_d, avg_u


    def get_opinions(self):
        opinions_in_network_T = [0, 0, 0]  # b, d, u
        opinions_in_network_F = [0, 0, 0]
        opinions_in_network_U = [0, 0, 0]

        for node in self.network.nodes(data=True):
            if node[1].get("expected_role") == 0:
                opinions_in_network_F[0] += node[1].get("b")
                opinions_in_network_F[1] += node[1].get("d")
                opinions_in_network_F[2] += node[1].get("u")

            elif node[1].get("expected_role") == 1:
                opinions_in_network_T[0] += node[1].get("b")
                opinions_in_network_T[1] += node[1].get("d")
                opinions_in_network_T[2] += node[1].get("u")

            else:
                opinions_in_network_U[0] += node[1].get("b")
                opinions_in_network_U[1] += node[1].get("d")
                opinions_in_network_U[2] += node[1].get("u")

        return opinions_in_network_T, opinions_in_network_F, opinions_in_network_U

    def get_opinions_plain(self):
        opinions_in_network_T = [0, 0, 0]  # b, d, u
        opinions_in_network_F = [0, 0, 0]
        opinions_in_network_U = [0, 0, 0]

        for node in self.network.nodes(data=True):
            if node[1].get("role") == 0:
                opinions_in_network_F[0] += node[1].get("b")
                opinions_in_network_F[1] += node[1].get("d")
                opinions_in_network_F[2] += node[1].get("u")

            elif node[1].get("role") == 1:
                opinions_in_network_T[0] += node[1].get("b")
                opinions_in_network_T[1] += node[1].get("d")
                opinions_in_network_T[2] += node[1].get("u")

            else:
                opinions_in_network_U[0] += node[1].get("b")
                opinions_in_network_U[1] += node[1].get("d")
                opinions_in_network_U[2] += node[1].get("u")

        return opinions_in_network_T, opinions_in_network_F, opinions_in_network_U

    def pick_TIP(self, n):
        r = 100
        s = 1
        w = 2

        b = r / (r + s + w)
        d = s / (r + s + w)
        u = w / (r + s + w)

        attrs = {str(n): {"role": 1, "b": b, "d": d, "u": u, "a": 1, "seed_or_not": 1, "expected_role": 1,
                          "free_node_or_not": 0}}
        nx.set_node_attributes(self.network, attrs)

        self.True_seeds.append((n))

    def pick_FIP(self, n):
        r = 1
        s = 100
        W = 2

        b = r / (r + s + W)
        d = s / (r + s + W)
        u = W / (r + s + W)

        attrs = {str(n): {"role": 0, "b": b, "d": d, "u": u, "a": 0, "seed_or_not": 1, "expected_role": 0,
                          "free_node_or_not": 0}}
        nx.set_node_attributes(self.network, attrs)

        self.False_seeds.append(n)

    '''
    # These are the old action functions
    def pick_unf_node(self):
        max_sum = 0
        for i in self.network.nodes:
            # if this node was picked by one party, cannot be chosen again.
            if self.network.nodes[i].get("seed_or_not") == 1 or self.network.nodes[i].get("role") != 2:
                continue

            uncertainty_neighbor_sum = 0
            for node in self.network.neighbors(i):
                if self.network.nodes[node].get("role") == 2:  # role == 2 is uncertainty node
                    uncertainty_neighbor_sum += 1

            if uncertainty_neighbor_sum > max_sum:
                max_sum = uncertainty_neighbor_sum
                self.unf_node = i

    def pick_rnf_node(self):
        max_sum = 0
        for i in self.network.nodes:
            if self.network.nodes[i].get("seed_or_not") == 1 or self.network.nodes[i].get("role") != 2:
                continue

            neighbor_reading_frequency = 0
            for node in self.network.neighbors(i):
                neighbor_reading_frequency = neighbor_reading_frequency + self.network.nodes[node].get("P_r")

            if neighbor_reading_frequency > max_sum:
                self.rnf_node = i

    def pick_snf_node(self):
        max_sum = 0
        for i in self.network.nodes:
            if self.network.nodes[i].get("seed_or_not") == 1 or self.network.nodes[i].get("role") != 2:
                continue

            neighbor_sharing_frequency = 0
            for node in self.network.neighbors(i):
                neighbor_sharing_frequency = neighbor_sharing_frequency + self.network.nodes[node].get("P_s")

            if neighbor_sharing_frequency > max_sum:
                self.snf_node = i
    '''

    # the following three action functions are the new design, but with the old name.
    # After we found the good design, we will change the names.
    def pick_unf_node(self):
        # Get nodes sorted by degree centrality in descending order
        nodes_degree = sorted(
            nx.degree_centrality(self.network).items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Track number of non-free nodes
        count = 0
        max_RS = 0
        
        # Iterate through sorted nodes
        for node, degree in nodes_degree:
            # Check if node is free
            if self.network.nodes[node].get("free_node_or_not") != 1:
                count += 1
                continue
            
            # Get sharing frequency
            sharing_freq = self.network.nodes[node].get("P_s")
            
            # Update max if higher
            if sharing_freq > max_RS:
                max_RS = sharing_freq
                self.unf_node = node
        
        # Set stop flag if all nodes checked were non-free
        return 1 if count >= self.node_density else 0
    
    def pick_unf_node_old(self):
        # RxS, active user first
        nodes_degree = nx.degree_centrality(self.network)
        count = 0
        max_RS = 0
        stop = 0
        while len(nodes_degree) > 0:
            max_key = max(nodes_degree, key=nodes_degree.get)
            if self.network.nodes[max_key].get("free_node_or_not") != 1:
                count += 1
                nodes_degree.pop(max_key)
                continue

            # RTimesS_frequency = self.network.nodes[i].get("P_r") * self.network.nodes[i].get("P_s")
            RTimesS_frequency = self.network.nodes[max_key].get("P_s")


            if RTimesS_frequency > max_RS:
                max_RS = RTimesS_frequency
                self.unf_node = max_key

            nodes_degree.pop(max_key)

        if count >= self.node_density:
            stop = 1

        return stop

    def pick_rnf_node(self):
        """
        Picks two nodes based on degree centrality:
        - rnf_node[0]: Attacker (neighbor has role == 1)
        - rnf_node[1]: Defender (neighbor has role == 0)

        Returns:
            int: 0 if both nodes are found, otherwise 1
        """
        # Calculate degree centrality once and sort nodes descendingly
        nodes_degree = sorted(
            nx.degree_centrality(self.network).items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        rnf_T_found = False  # Indicates if attacker node is found
        rnf_F_found = False  # Indicates if defender node is found
        
        for node, degree in nodes_degree:
            if self.network.nodes[node].get("free_node_or_not") != 1:
                continue  # Skip non-free nodes
            
            neighbors = self.network.neighbors(node)
            roles = set(self.network.nodes[neighbor].get("role") for neighbor in neighbors)
            
            if not rnf_F_found and 0 in roles:
                self.rnf_node[1] = node  # Defender
                rnf_F_found = True
            
            if not rnf_T_found and 1 in roles:
                self.rnf_node[0] = node  # Attacker
                rnf_T_found = True
            
            # If both nodes are found, stop the search
            if rnf_T_found and rnf_F_found:
                return 0  # Successfully found both nodes
        
        # If either attacker or defender is not found
        return 1
    

    def pick_rnf_node_old(self):
        # blocking
        nodes_degree = nx.degree_centrality(self.network)
        # print(nodes_degree)
        stop = 1
        # to count if both rnf nodes are found
        rnf_T = 0
        rnf_F = 0
        while len(nodes_degree) > 0:
            max_key = max(nodes_degree, key=nodes_degree.get)

            if self.network.nodes[max_key].get("free_node_or_not") == 1:
                for node in self.network.neighbors(max_key):
                    # if this neighbor is in F party
                    if self.network.nodes[node].get("role") == 0 and rnf_T == 0:
                        # rnf_node[0] is for attacker action, [1] is for defender
                        self.rnf_node[1] = max_key
                        rnf_T = 1
                    # if this neighbor is in T party
                    if self.network.nodes[node].get("role") == 1 and rnf_F == 0:
                        self.rnf_node[0] = max_key
                        rnf_F = 1
            nodes_degree.pop(max_key)
            if rnf_T == 1 and rnf_F == 1:
                stop = 0
                break
        return stop

    # def pick_rnf_node(self):
    #     # I want to do uncertain neighbor first
    #     count = 0
    #     max_U = 0
    #     stop = 0
    #     for i in self.network.nodes:
    #         if self.network.nodes[i].get("free_node_or_not") != 1:
    #             count += 1
    #             continue
    #
    #         under_uncertainty = self.network.nodes[i].get("u")
    #
    #         if under_uncertainty > max_U:
    #             max_U = under_uncertainty
    #             self.rnf_node[0] = i
    #             self.rnf_node[1] = i
    #     if count >= self.node_density:
    #         stop = 1
    #
    #     return stop

    def pick_snf_node(self):
        """
        Picks the node with the maximum two-hop influence among free nodes.

        Returns:
            int: 0 if a suitable node is found, otherwise 1
        """
        # 获取所有自由节点
        free_nodes = [n for n, data in self.network.nodes(data=True) if data.get("free_node_or_not") == 1]
        
        # 如果没有自由节点，设置 stop 为 1 并返回
        if not free_nodes:
            return 1  # stop = 1
        
        free_nodes_set = set(free_nodes)
        
        # 预先计算每个自由节点的自由邻居数量
        free_neighbor_counts = {
            n: sum(1 for neighbor in self.network.neighbors(n) if neighbor in free_nodes_set)
            for n in free_nodes
        }
        
        # 初始化最大影响力和选定的节点
        max_two_hops_influence = -1
        selected_node = None
        
        # 遍历每个自由节点，计算其两跳影响力
        for node in free_nodes:
            influence = sum(free_neighbor_counts[neighbor] + 1 for neighbor in self.network.neighbors(node) if neighbor in free_nodes_set)
            if influence > max_two_hops_influence:
                max_two_hops_influence = influence
                selected_node = node
        
        # 如果找到了合适的节点，设置 snf_node 并返回 0
        if selected_node:
            self.snf_node = selected_node
            return 0  # stop = 0
        else:
            return 1  # stop = 1
    
    def pick_snf_node_old(self):
        # sub-greedy
        count = 0
        stop = 0
        max_two_hops_influence = 0
        for i in self.network.nodes:
            if self.network.nodes[i].get("free_node_or_not") != 1:
                count += 1
                continue

            else:
                # only look at FREE node
                free_neighbor = 0
                for j in self.network.neighbors(i):
                    if self.network.nodes[j].get("free_node_or_not") == 1:
                        free_neighbor = free_neighbor + self.get_number_of_free_neighbors(j) + 1
                if free_neighbor > max_two_hops_influence:
                    max_two_hops_influence = free_neighbor
                    self.snf_node = i
        if count >= self.node_density:
            stop = 1
        return stop

    def pick_cf_node(self):
        # degree
        # cf
        nodes_degree = nx.degree_centrality(self.network)
        # print(nodes_degree)
        stop = 1
        while len(nodes_degree) > 0:
            max_key = max(nodes_degree, key=nodes_degree.get)
            self.cf_node = max_key
            nodes_degree.pop(max_key)
            if self.network.nodes[self.cf_node].get("free_node_or_not") == 1:
                stop = 0
                break
        return stop

    def pick_ev_node(self):
        # eigenvector
        # ev
        nodes_ev = nx.eigenvector_centrality(self.network)
        stop = 1
        while len(nodes_ev) > 0:
            max_key = max(nodes_ev, key=nodes_ev.get)
            self.snf_node = max_key
            nodes_ev.pop(max_key)
            # if self.network.nodes[self.snf_node].get("seed_or_not") != 1 and self.network.nodes[self.snf_node].get("role") == 2:
            if self.network.nodes[self.snf_node].get("free_node_or_not") == 1:
                stop = 0
                break
        # print("stop : ", stop)
        return stop

    def pick_random_node(self):
        while True:
            # self.random_node = random.choice(list(self.network.nodes))
            self.random_node = random.choice([self.unf_node, self.rnf_node[0], self.rnf_node[1], self.snf_node, self.cf_node])
            if self.network.nodes[self.random_node].get("free_node_or_not") == 1:
                break

    def pick_greedy_node(self):
        pass

    def opinion_update(self, node_i, node_j, accepting_prob, uncertainty_maximization_threshold):
        # i will update opinion when encounter j
        # action_or_not = [0, 1]
        # if no filtering, always accept, so we set this value to 1.
        accepting_action = 1
        node_type_i = self.network.nodes[node_i].get("node_type")
        # print(f"Node type of {node_i} is {node_type_i}")

        '''Comment out to disable Uncertainty Maximization'''
        if node_type_i == 0:  # only apply UM when using UOM
            b_i = self.network.nodes[node_i].get("b")
            d_i = self.network.nodes[node_i].get("d")
            u_i = self.network.nodes[node_i].get("u")  # low vacuity

            dissonance_ij = b_i + d_i - abs(b_i - d_i)  # high disonass
            if u_i < uncertainty_maximization_threshold[0]:
                if dissonance_ij >= uncertainty_maximization_threshold[1]:  # like b = 0.3, d = 0.6, diss = 0.59....
                    # print("Uncertainty Maximization triggered")
                    # print("u_i before is ", u_i)

                    # only when they haven't belonged to any party yet, we use the UM
                    self.uncertainty_maximization(node_i)
                    # print("u_i after is ", self.network.nodes[node_i].get("u"))
                    # print("current UM times: ", self.UM_trigger_times)

        # print(f"Node {node_i} is updating because node {node_j} is sharing")

        b_i = self.network.nodes[node_i].get("b")
        d_i = self.network.nodes[node_i].get("d")
        u_i = self.network.nodes[node_i].get("u")
        a_i = self.network.nodes[node_i].get("a")
        # print("for receiver: ", b_i, d_i, u_i)

        b_j = self.network.nodes[node_j].get("b")
        d_j = self.network.nodes[node_j].get("d")
        u_j = self.network.nodes[node_j].get("u")
        a_j = self.network.nodes[node_j].get("a")
        # print("for sender: ", b_j, d_j, u_j)

        '''Choose the Opinion Model'''
        # UOM
        uc_ij = (1 - u_i) * (1 - u_j)
        # HOM
        denominator = math.sqrt(b_i * b_i + d_i * d_i) * math.sqrt(b_j * b_j + d_j * d_j)
        if denominator <= 0.0001:
            hc_ij = 1
        else:
            hc_ij = (b_i * b_j + d_i * d_j) / (math.sqrt(b_i * b_i + d_i * d_i) * math.sqrt(b_j * b_j + d_j * d_j))

        # opinion_model = self.opinion_model_selection(node_type=node_type_i)
        if node_type_i == 0:
            c_ij = uc_ij
        elif node_type_i == 1:
            c_ij = hc_ij
        else:  # this is no trust based
            c_ij = 1
        '''End Opinion Model Choice'''

        beta = 1 - c_ij * (1 - u_i) * (1 - u_j)
        if u_i <= 0.001 or beta <= 0.0001:  # if user i has very low u value, i won't update
            continue_spread_1 = 0
        else:
            continue_spread_1 = 1

            if accepting_action == 1:
                new_b = (b_i * (1 - c_ij * (1 - u_j)) + c_ij * b_j * u_i) / beta
                new_d = (d_i * (1 - c_ij * (1 - u_j)) + c_ij * d_j * u_i) / beta
                new_u = (u_i * (1 - c_ij * (1 - u_j))) / beta
                new_a = ((a_i - (a_i + a_j) * u_i) * (1 - c_ij * (1 - u_j)) + a_j * u_i) / (
                            beta - u_i * (1 - c_ij * (1 - u_j)))
                # print(f"previous opinion of {node_i} is b: {b_i}, d: {d_i}, u: {u_i}")
                # print(f"after opinion of {node_i} is b: {new_b}, d: {new_d}, u: {new_u}")
                new_P_b = new_b + new_a * new_u
                new_P_d = new_d + (1 - new_a) * new_u
                # print("after update: ", new_b, new_d, new_u)
                # print("the expected value: ", new_P_b, new_P_d)

                if new_b > 0.5:
                    new_role = 1
                elif new_d > 0.5:
                    new_role = 0
                else:
                    new_role = 2
                # After propagation, we use expected b and d to judge whether a user is T or F
                if new_P_b > 0.5:
                    new_expected_role = 1
                elif new_P_d > 0.5:
                    new_expected_role = 0
                else:
                    new_expected_role = 2

                if new_u > 0.5:
                    free_node_or_not = 1
                else:
                    free_node_or_not = 0

                # print(f"original: {b_i},{d_i},{u_i},{a_i}.\n new: {new_b},{new_d},{new_u},{new_a}")
                attrs = {node_i: {"role": new_role, "b": new_b, "d": new_d, "u": new_u, "a": new_a,
                                  "free_node_or_not": free_node_or_not, "expected_role": new_expected_role}}
                nx.set_node_attributes(self.network, attrs)
                # list(G.nodes(data = True))

            else:
                # action is 0, no update
                pass

        return continue_spread_1 * accepting_action

    '''Node u has different sharing behaviors to its neighbors'''

    def update_after_pick_seed(self, start_node, uncertainty_maximization_threshold):
        action_or_not = [0, 1]  # [0, 1] takes action according to sharing prob, [1. 1] always takes action.

        for node in self.network.nodes:
            attrs = {node: {"color": "white"}}
            nx.set_node_attributes(self.network, attrs)

        attrs = {start_node: {"color": "black"}}
        nx.set_node_attributes(self.network, attrs)
        q = list()
        q.append(start_node)
        # print("start node is : ", start_node)

        while len(q) > 0:
            node_u = q.pop(0)
            # print(f"pop out {node_u}")
            sharing_prob = self.network.nodes[node_u].get("P_s")
            for node in self.network.neighbors(node_u):
                # print("check " + node_u + "'s neighbor node: " + node)

                # the TIP and FIP won't change their opinion
                seed_node_or_not = self.network.nodes[node].get("seed_or_not")
                if seed_node_or_not == 1:
                    # print(node + " is a seed, skip.")
                    continue

                if self.network.nodes[node].get("color") == "white":
                    # sharing_prob = self.network.get_edge_data(node_u, node).get("P_s")
                    # this sharing behavior is for node_u, so we check if it shares when we just pop it out.
                    reading_prob = self.network.nodes[node].get("P_r")
                    accepting_prob = self.network.nodes[node].get("P_f")
                    # print("accepting_prob: ", accepting_prob)

                    # need to discuss where to put this sharing action, for each node or for node_u itself
                    # if we check the sharing action here, node_u has different sharing behavior to its neighbors
                    sharing_action = \
                        random.choices(action_or_not, weights=(100 * (1 - sharing_prob), 100 * sharing_prob), k=1)[0]
                    reading_action = \
                        random.choices(action_or_not, weights=(100 * (1 - reading_prob), 100 * reading_prob), k=1)[0]

                    if sharing_action:
                        if reading_action:
                            continue_spread = self.opinion_update(node, node_u, accepting_prob,
                                                                  uncertainty_maximization_threshold)
                            # if node has very low u value, it won't update, continue_spread = 0
                            # if the accepting action is 0, it also won't update, continue_spread = 0

                            if continue_spread == 1:
                                # print("shared from " + node_u + " to " + node)
                                attrs = {node: {"color": "black"}}
                                nx.set_node_attributes(self.network, attrs)
                                q.append(node)
                                # print(f"push in {node}")
                            else:
                                pass
                                # print("don't share from " + node_u + " to " + node)
                                # no more spread from this neighbor node because his u is so low.

                        else:
                            pass
                            # print(f"didn't push in {node}")
                    else:
                        pass
                        # print(f"didn't push in {node}")
                else:
                    pass
                    # print(f"{node} has been visited.")
        # print("UM times: ", self.UM_trigger_times)
        return self.get_UM_triggered_time()

    def random_sharing(self):
        # each edge share once
        action_or_not = [0, 1]  # [0, 1] takes action according to sharing prob, [1. 1] always takes action.
        for edge in self.network.edges(data=True):
            node_u = edge[0]
            node_v = edge[1]

            sharing_prob_u = self.network.nodes[node_u].get("P_s")
            sharing_prob_v = self.network.nodes[node_v].get("P_s")
            reading_prob_u = self.network.nodes[node_u].get("P_r")
            reading_prob_v = self.network.nodes[node_v].get("P_r")
            accept_prob_u = self.network.nodes[node_u].get("P_f")
            accept_prob_v = self.network.nodes[node_v].get("P_f")
            # print("sharep_u: ", sharing_prob_u)
            # print("readp_u: ", reading_prob_u)
            # print("sharep_v: ", sharing_prob_v)
            # print("readp_v: ", reading_prob_v)

            # whether user u share the info to v, so far it is only for undirected graph, will be using directed graph.
            # random.choices returns a list, even though we only ask one element by setting k=1
            sharing_action_u = random.choices(action_or_not, weights=(100 * (1 - sharing_prob_u), 100 * sharing_prob_u), k=1)[0]
            sharing_action_v = \
            random.choices(action_or_not, weights=(100 * (1 - sharing_prob_v), 100 * sharing_prob_v), k=1)[0]
            # though user's reading behavior only relates to themselves, they may choose to read or not when considering different edges
            # That's way we calculate reading_action for every edge.
            reading_action_u = \
            random.choices(action_or_not, weights=(100 * (1 - reading_prob_u), 100 * reading_prob_u), k=1)[0]
            reading_action_v = \
            random.choices(action_or_not, weights=(100 * (1 - reading_prob_v), 100 * reading_prob_v), k=1)[0]

            # print(f"the edge is {node_v},{node_u}, sharing prob v is {sharing_prob_v}, reading prob u is {reading_prob_u}")
            # print(f"sharing action v id {sharing_action_v}, reading action u is {reading_action_u}")
            if sharing_action_v and reading_action_u:  # both are 1, means v shares with u, u reads as well
                # print("v share and u read")
                self.opinion_update(node_u, node_v, accepting_prob=1, uncertainty_maximization_threshold=[0.001, 0.6])
            else:
                pass

            # print(f"the edge is {node_u},{node_v}, sharing prob u is {sharing_prob_u}, reading prob v is {reading_prob_v}")
            # print(f"sharing action u id {sharing_action_u}, reading action v is {reading_action_v}")
            if sharing_action_u and reading_action_v:  # both are 1, means u shares with v, v reads as well
                # print("u share and v read")
                self.opinion_update(node_v, node_u, accepting_prob=1, uncertainty_maximization_threshold=[0.001, 0.6])
            else:
                pass

            # if sharing_action_v and reading_action_u:  # both are 1, means v shares with u, u reads as well
            #     print("v share and u read")
            #     a_random_number = random.random()
            #     # if u's False information filtering ability is less than this random value, u will accept v's opinion and update
            #     if a_random_number > accept_prob_u:
            #         print("u update")
            #         self.opinion_update(node_u, node_v, accepting_prob=1, uncertainty_maximization_threshold=0)


    def number_to_level(self, node_number, max_value):
        if node_number >= max_value * 3 / 4:
            node_number_level = 3
        elif node_number >= max_value / 2:
            node_number_level = 2
        elif node_number >= max_value / 4:
            node_number_level = 1
        else:
            node_number_level = 0
        return node_number_level

    '''Node u has the same sharing behavior to all its neighbors'''

    def update_after_pick_seed_1(self, start_node):
        action_or_not = [1, 1]  # change to 1, 1 for always taking action

        for node in self.network.nodes:
            attrs = {node: {"color": "white"}}
            nx.set_node_attributes(self.network, attrs)

        attrs = {start_node: {"color": "black"}}
        nx.set_node_attributes(self.network, attrs)
        q = list()
        q.append(start_node)

        while len(q) > 0:
            node_u = q.pop(0)
            # print(f"pop out {node_u}")
            sharing_prob = self.network.nodes[node_u].get("P_s")
            # if we check the sharing action here, node_u has same sharing behavior to its neighbors.
            sharing_action = \
                random.choices(action_or_not, weights=(100 * (1 - sharing_prob), 100 * sharing_prob), k=1)[0]
            if sharing_action == 0:
                continue

            for node in self.network.neighbors(node_u):

                # the TIP and FIP won't change their opinion
                seed_or_not = self.network.nodes[node].get("seed_or_not")
                if seed_or_not == 1:
                    continue

                if self.network.nodes[node].get("color") == "white":
                    # sharing_prob = self.network.get_edge_data(node_u, node).get("P_s")
                    # this sharing behavior is for node_u, so we check if it shares when we just pop it out.
                    # sharing_prob = self.network.nodes[node].get("P_s")
                    reading_prob = self.network.nodes[node].get("P_r")
                    accepting_prob = self.network.nodes[node].get("P_f")

                    # need to discuss where to put this sharing action, for each node or for node_u itself
                    reading_action = \
                        random.choices(action_or_not, weights=(100 * (1 - reading_prob), 100 * reading_prob), k=1)[0]
                    # a_random_number = random.random()
                    a_random_number = 1
                    if a_random_number >= accepting_prob:
                        accepting_action = 1
                    else:
                        accepting_action = 0

                    if reading_action:
                        if accepting_action:
                            self.opinion_update(node, node_u)
                            attrs = {node: {"color": "black"}}
                            nx.set_node_attributes(self.network, attrs)
                            q.append(node)
                            # print(f"push in {node}")
                            # read and accept
                        else:
                            pass
                            # print(f"didn't push in {node}")
                            # read but not accept, might trigger defense
                    else:
                        pass
                        # print(f"didn't push in {node}")
                        # don't read
                else:
                    pass
                    # print(f"didn't push in {node}")
                    # this node has updated his opinion

    def uncertainty_maximization(self, node):
        b = self.network.nodes[node].get("b")
        d = self.network.nodes[node].get("d")
        u = self.network.nodes[node].get("u")
        a = self.network.nodes[node].get("a")

        P_b = b + a * u
        P_d = d + (1 - a) * u

        if P_b / a > P_d / (1 - a):
            new_u = P_d / (1 - a)
        else:
            new_u = P_b / a
        new_b = P_b - a * new_u
        new_d = P_d - (1 - a) * new_u
        new_a = a
        new_P_b = new_b + new_a * new_u
        new_P_d = new_d + (1 - new_a) * new_u

        if new_b > 0.5:
            new_role = 1
        elif new_d > 0.5:
            new_role = 0
        else:
            new_role = 2
        # After propagation, we use expected b and d to judge whether a user is T or F
        if new_P_b > 0.5:
            new_expected_role = 1
        elif new_P_d > 0.5:
            new_expected_role = 0
        else:
            new_expected_role = 2

        if new_u > 0.5:
            free_node_or_not = 1
        else:
            free_node_or_not = 0

        # print(f"original: {b_i},{d_i},{u_i},{a_i}.\n new: {new_b},{new_d},{new_u},{new_a}")
        attrs = {node: {"role": new_role, "b": new_b, "d": new_d, "u": new_u, "a": new_a,
                        "free_node_or_not": free_node_or_not, "expected_role": new_expected_role}}
        nx.set_node_attributes(self.network, attrs)

        self.UM_trigger_times += 1

    def check_bdu_sum(self):
        # Verify if b, d and u add up to 1, do it after the opinion updating
        count = 0
        for node in self.network:
            b = self.network.nodes[node].get("b")
            d = self.network.nodes[node].get("d")
            u = self.network.nodes[node].get("u")
            check_sum = b + d + u
            if abs(check_sum - 1) > 1e-4:
                print("Error in node ID: ", node)
                print("The wrong check sum is: ", check_sum)
                print("node attributes: ", )
                raise Exception("Sorry, checksum check didn't pass")
            else:
                count += 1
        if count == self.network.number_of_nodes():
            # print("Check Pass! Sum of b, d, and u for all nodes are 1")
            pass

    def plot_all_opinions_prep(self):
        b = []
        d = []
        u = []
        # users opinion in T party
        b_T = []
        d_T = []
        u_T = []
        # users opinion in F party
        b_F = []
        d_F = []
        u_F = []
        # users opinion in Uncertain
        b_U = []
        d_U = []
        u_U = []
        padding_item = 0

        for node in self.network.nodes(data=True):
            b.append(node[1].get('b'))
            d.append(node[1].get('d'))
            u.append(node[1].get('u'))

            # if a user in F party, we gather his opinion
            if node[1].get('expected_role') == 0:
                # if node[1].get('role') == 0:
                b_F.append(node[1].get('b'))
                d_F.append(node[1].get('d'))
                u_F.append(node[1].get('u'))
            # we need to padding because when draw the opinion, it has to be a two dimensional arrays
            else:
                b_F.append(padding_item)
                d_F.append(padding_item)
                u_F.append(padding_item)

            # if a user in T party
            if node[1].get('expected_role') == 1:
                # if node[1].get('role') == 1:
                b_T.append(node[1].get('b'))
                d_T.append(node[1].get('d'))
                u_T.append(node[1].get('u'))
            else:
                b_T.append(padding_item)
                d_T.append(padding_item)
                u_T.append(padding_item)

            # if a user in Uncertain party
            if node[1].get('expected_role') == 2:
                # if node[1].get('role') == 2:
                b_U.append(node[1].get('b'))
                d_U.append(node[1].get('d'))
                u_U.append(node[1].get('u'))
            else:
                b_U.append(padding_item)
                d_U.append(padding_item)
                u_U.append(padding_item)

        self.arr_b_T.append(b_T)
        self.arr_d_T.append(d_T)
        self.arr_u_T.append(u_T)

        self.arr_b_F.append(b_F)
        self.arr_d_F.append(d_F)
        self.arr_u_F.append(u_F)

        self.arr_b_U.append(b_U)
        self.arr_d_U.append(d_U)
        self.arr_u_U.append(u_U)

        self.arr_b.append(b)
        self.arr_d.append(d)
        self.arr_u.append(u)
        self.plot_all_opinions_times.append(len(self.plot_all_opinions_times))

    # def plot_all_users_opinions(self):
    #     self.plot_opinions_func(self.arr_b, self.arr_d, self.arr_u, title=train_combination + "_all users opinion")
    #     # self.plot_opinions_func(self.arr_b_T, self.arr_d_T, self.arr_u_T, title=train_combination + "_T party users opinion")
    #     # self.plot_opinions_func(self.arr_b_F, self.arr_d_F, self.arr_u_F, title=train_combination + "_F party users opinion")
    #     # self.plot_opinions_func(self.arr_b_U, self.arr_d_U, self.arr_u_U, title=train_combination + "_Uncertain users opinion")

    def plot_opinions_func(self, b, d, u, title):
        numbers = self.count_numbers()
        print(f"True: {numbers[1]}, False: {numbers[0]}, Uncertain: {numbers[2]}")

        colors = ['blue', 'red', 'green', 'gray']
        # blue: b, red: d, green: u, gray: a
        # fig, ax1, ax2, ax3 = plt.subplots(3)

        plt.suptitle(title)

        plt.subplot(3, 1, 1)
        plt.plot(self.plot_all_opinions_times, b, 'o', color=colors[0], markersize=1, linestyle='')
        plt.title('b')

        plt.subplot(3, 1, 2)
        plt.plot(self.plot_all_opinions_times, d, 'o', color=colors[1], markersize=1, linestyle='')
        plt.title('d')

        plt.subplot(3, 1, 3)
        plt.plot(self.plot_all_opinions_times, u, 'o', color=colors[2], markersize=1, linestyle='')
        plt.title('u')

        # plt.plot(update_times, arr_a, color = colors[3])
        plt.xlabel('time')

        # make directory for saving figures
        figures_dir = "Individual users opinion"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        fig_save_path = figures_dir + '/' + title + '.png'
        plt.savefig(fig_save_path)

        # show a legend on the plot
        # plt.legend()
        plt.show()

    def plot_average_opinions_prep(self):
        T_party, F_party, U_party = self.get_opinions()

        b = T_party[0] + F_party[0] + U_party[0]
        average_b = b / self.node_density
        d = T_party[1] + F_party[1] + U_party[1]
        average_d = d / self.node_density
        u = T_party[2] + F_party[2] + U_party[2]
        average_u = u / self.node_density

        self.average_bs.append(average_b)
        self.average_ds.append(average_d)
        self.average_us.append(average_u)
        self.plot_average_opinion_times.append(len(self.plot_average_opinion_times))

    def plot_average_opinions(self):
        plt.plot(self.plot_average_opinion_times, self.average_bs, label="b")
        plt.plot(self.plot_average_opinion_times, self.average_ds, label="d")
        plt.plot(self.plot_average_opinion_times, self.average_us, label="u")
        plt.xlabel('time')
        plt.ylabel('average opinion')
        plt.title('Average Opinion Value over the whole network')
        # show a legend on the plot
        plt.legend()
        plt.show()

    def get_UM_triggered_time(self):
        # print("Uncertainty Maximization is triggered: " + str(self.UM_trigger_times) + " times.")
        return self.UM_trigger_times

    def log_all_opinions(self):
        log_opinion_propagation_test = "opinion propagation.csv"
        print("logging at : " + log_opinion_propagation_test)
        # haven't finished yet
# sys.stdout.close()