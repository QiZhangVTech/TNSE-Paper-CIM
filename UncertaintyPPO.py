import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from torch.distributions import Categorical
from PPO_UM_helper import RelativeMassBalance

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################ Model Opinion ################################\
class Model_Opinions:
    def __init__(self):
        self.old_opinions = torch.tensor([0.25, 0.25, 0.25, 0.25, 0])
        # self.new_opinions = torch.tensor([0.25, 0.25, 0.25, 0.25, 0])
        self.uncertainty_threshold = torch.tensor(0.9)
        self.dissonance_threshold = torch.tensor(0.7)
        # pick vacuity thre as 0.5, try diss 0.1, 0.3, 0.5, 0.7, 0.9
        self.a1 = torch.tensor(0.25)
        self.a2 = torch.tensor(0.25)
        self.a3 = torch.tensor(0.25)
        self.a4 = torch.tensor(0.25)

    def calculateUM(self, model_opinion):
        b1 = model_opinion[0]
        b2 = model_opinion[1]
        b3 = model_opinion[2]
        b4 = model_opinion[3]
        if len(model_opinion) == 4:
            u = torch.tensor(0).to(device)
            # self.a1, self.a2, self.a3, self.a4 = self.calculate_base_rates(self.old_opinions)
            # self.a1 = b1
            # self.a2 = b2
            # self.a3 = b3
            # self.a4 = b4
        else:
            u = model_opinion[4]

        # print("length of opinion: ", len(model_opinion))
        # print("b1: ", b1)
        # print("b2: ", b2)
        # print("b3: ", b3)
        # print("b4: ", b4)
        # print("u: ", u)
        # print("Check opinion sum, b1 + b2 + b3 + b4 + u = ", b1 + b2 + b3 + b4 + u)
        # print("a1: ", self.a1)
        # print("a2: ", self.a2)
        # print("a3: ", self.a3)
        # print("a4: ", self.a4)
        # print("old opinion: ", self.old_opinions)

        # Pb = b + a * u
        # Uncertainty Maximization: u' = min(Pb_i/a_i), b'_i = Pb_i - a_i * u'
        u_UM = min((b1/self.a1 + u), (b2/self.a2 + u), (b3/self.a3 + u), (b4/self.a4 + u))
        b1_UM = b1 - self.a1 * u_UM
        b2_UM = b2 - self.a2 * u_UM
        b3_UM = b3 - self.a3 * u_UM
        b4_UM = b4 - self.a4 * u_UM
        # print("Before deal with calculation error: ")
        # print("b1_UM: ", b1_UM)
        # print("b2_UM: ", b2_UM)
        # print("b3_UM: ", b3_UM)
        # print("b4_UM: ", b4_UM)
        # print("u_UM: ", u_UM)
        # print("Check UM opinion sum, b1_UM + b2_UM + b3_UM + b4_UM + u_UM = ", b1_UM + b2_UM + b3_UM + b4_UM + u_UM)
        if abs(b1_UM - 0) < 1e-4:
        # if b1_UM < 0:
            b1_UM = torch.tensor(0).to(device)
        if abs(b2_UM - 0) < 1e-4:
        # if b2_UM < 0:
            b2_UM = torch.tensor(0).to(device)
        if abs(b3_UM - 0) < 1e-4:
        # if b3_UM < 0:
            b3_UM = torch.tensor(0).to(device)
        if abs(b4_UM - 0) < 1e-4:
        # if b4_UM < 0:
            b4_UM = torch.tensor(0).to(device)
        # print("After deal with calculation error: ")
        # print("b1_UM: ", b1_UM)
        # print("b2_UM: ", b2_UM)
        # print("b3_UM: ", b3_UM)
        # print("b4_UM: ", b4_UM)
        # print("u_UM: ", u_UM)
        # print("Check UM opinion sum, b1_UM + b2_UM + b3_UM + b4_UM + u_UM = ", b1_UM + b2_UM + b3_UM + b4_UM + u_UM)
        self.old_opinions = torch.tensor([b1_UM.item(), b2_UM.item(), b3_UM.item(), b4_UM.item(), u_UM.item()])
        # print("old opinion updated: ", self.old_opinions)
        # return u_UM
        return self.old_opinions
        # return torch.tensor([b1_UM.item(), b2_UM.item(), b3_UM.item(), b4_UM.item(), u_UM.item()])

    def calculateDissonance(self, model_opinion):
        b1 = model_opinion[0]
        b2 = model_opinion[1]
        b3 = model_opinion[2]
        b4 = model_opinion[3]
        diss = (b1 * (b2 * RelativeMassBalance(b2, b1) + b3 * RelativeMassBalance(b3, b1) + b4 * RelativeMassBalance(b4, b1)))/(b2 + b3 + b4) \
               + (b2 * (b1 * RelativeMassBalance(b1, b2) + b3 * RelativeMassBalance(b3, b2) + b4 * RelativeMassBalance(b4, b2)))/(b1 + b3 + b4) \
               + (b3 * (b1 * RelativeMassBalance(b1, b3) + b2 * RelativeMassBalance(b2, b3) + b4 * RelativeMassBalance(b4, b3)))/(b1 + b2 + b4) \
               + (b4 * (b1 * RelativeMassBalance(b1, b4) + b2 * RelativeMassBalance(b2, b4) + b3 * RelativeMassBalance(b3, b4)))/(b1 + b2 + b3)
        return diss

    def opinion2evidence(self, model_opinion):
        evidences = []
        W = 4  # W is the number of class
        b1 = model_opinion[0]
        b2 = model_opinion[1]
        b3 = model_opinion[2]
        b4 = model_opinion[3]
        u = model_opinion[4]

        # Formula 3.23 in SL book, r_i = W*b_i/u
        r1 = W * b1 / u
        r2 = W * b2 / u
        r3 = W * b3 / u
        r4 = W * b4 / u
        evidences = torch.tensor([r1, r2, r3, r4])
        return evidences

    def evidence2opinion(self, evidence):
        r1 = evidence[0]
        r2 = evidence[1]
        r3 = evidence[2]
        r4 = evidence[3]

        b1 = r1 / (r1 + r2 + r3 + r4)
        b2 = r2 / (r1 + r2 + r3 + r4)
        b3 = r3 / (r1 + r2 + r3 + r4)
        b4 = r4 / (r1 + r2 + r3 + r4)
        opinions = torch.tensor([b1.item(), b2.item(), b3.item(), b4.item(), 0])
        return opinions


    def calculate_base_rates(self, model_opinion):
        b1 = model_opinion[0]
        b2 = model_opinion[1]
        b3 = model_opinion[2]
        b4 = model_opinion[3]
        if b1 < 1e-4: b1 = 1e-4
        if b2 < 1e-4: b2 = 1e-4
        if b3 < 1e-4: b3 = 1e-4
        if b4 < 1e-4: b4 = 1e-4

        # Formula 3.23 in SL book, r_i = W*b_i/u
        # base rate a_i = r_i/sum of r_i = b_i/sum of b_i
        a1 = b1 / (b1 + b2 + b3 + b4)
        a2 = b2 / (b1 + b2 + b3 + b4)
        a3 = b3 / (b1 + b2 + b3 + b4)
        a4 = b4 / (b1 + b2 + b3 + b4)
        return a1, a2, a3, a4


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.evidences = []
        self.uncertainties = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.evidences[:]
        del self.uncertainties[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.opinion_model = Model_Opinions()

    def act(self, state):
        raw_actor_output = self.actor(state)
        evidences = F.relu(raw_actor_output) + 1  # Ensure non-negative outputs and shift to make Dirichlet parameters
        # print("Evidence is: ", evidences)
        action_probs = F.softmax(raw_actor_output, dim=-1)  # Action
        # print("action probs: ", action_probs)

        old_opinions_after_UM = self.opinion_model.calculateUM(model_opinion=action_probs)
        # dissonance = self.opinion_model.calculateDissonance(model_opinion=old_opinions_after_UM)
        dissonance = self.opinion_model.calculateDissonance(model_opinion=action_probs)
        uncertainty = old_opinions_after_UM[4]
        # print("Dissonance: ", dissonance)
        # print("uncertainty: ", uncertainty)
        # print("Opinion after UM is: ", old_opinions_after_UM)

        # opinion2evidence = self.opinion_model.opinion2evidence(self.opinion_model.old_opinions)
        # print("Opinion after UM to Evidence is: ", opinion2evidence)

        dist = Categorical(action_probs)
        dist_entropy = dist.entropy()
        max_entropy = Categorical(probs=torch.ones_like(action_probs) / len(action_probs)).entropy()
        # Normalize the entropy
        normalized_entropy = dist_entropy / max_entropy

        # print("action probs are: ", action_probs)
        # print("dist is categorical: ", dist)
        explore = 0
        if uncertainty > self.opinion_model.uncertainty_threshold:
            action = random.choice(torch.tensor([0, 1, 2, 3]).to(device))
            explore = 1
            # print("High vacuity")
            # print("exploration action: ", action)
        else:
        #     print("Low vacuity")
            if dissonance > self.opinion_model.dissonance_threshold:
                action = random.choice(torch.tensor([0, 1, 2, 3]).to(device))
                explore = 1
                # print("High dissonance")
                # print("exploration action: ", action)
                # new_opinion = self.opinion_model.calculateUM(model_opinion=old_opinions_after_UM)
                # new_dissonance = self.opinion_model.calculateDissonance(model_opinion=new_opinion)
                # if new_dissonance < self.opinion_model.dissonance_threshold:
                #     action = dist.sample()
                #     print("High dissonance and low diss after UM")
                #     print("exploiation action: ", action)
                # else:
                #     action = random.choice(torch.tensor([0, 1, 2, 3]).to(device))
                #     print("High dissonance and high diss after UM")
                #     print("exploration action: ", action)
            else:
                action = dist.sample()
                # print("Low dissonance")
                # print("exploiation action: ", action)


        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach(), action_probs, old_opinions_after_UM, uncertainty, dissonance, normalized_entropy, explore

    def evaluate(self, state, action):
        raw_actor_output = self.actor(state)
        action_probs = F.softmax(raw_actor_output, dim=-1)  # Action
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        # return action_logprobs, state_values, uncertainty
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val, action_probs, opinion_after_UM, uncertainty, dissonance, entropy, explore = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item(), np.array(action_probs.tolist()), np.array(opinion_after_UM.tolist()), uncertainty.item(), dissonance.item(), entropy.item(), explore

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # print("update agent")
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # logprobs, state_values, uncertainty = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)
            # print(f"The three loss: surr1 is {surr1}, surr2 is {surr2}, mesloss is {self.MseLoss(state_values, rewards)}, entropy is {dist_entropy}")
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * uncertainty

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))