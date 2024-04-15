#!/usr/bin/env python
# coding: utf-8

# In[1]:


from abc import ABC, abstractmethod
from logs import *
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


# In[36]:


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.reward = reward     # True reward
        self.p = p  # Initialize bandit options
        self.learning_process = []    # List of learning process 
        self.N = np.zeros(len(p))  # Number of times each arm has been pulled
        self.bandits = None    # List of bandits
        self.data = []  # List to store experiment data
        self.Q = np.zeros(len(p))  # Estimated reward for each arm
        self.rewards = []  # Rewards obtained during the experiment
        self.regrets = []  # Regret accumulated during the experiment

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self, bandit_name, rewards, algorithm):
        # Store data in CSV
        with open('bandit_rewards.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for reward in rewards:
                writer.writerow([bandit_name, reward, algorithm])

        # Log average reward
        avg_reward = sum(rewards) / len(rewards)
        logger.info(f'Average reward for {bandit_name} using {algorithm}: {avg_reward}')

        # Log average regret (assuming regret is tracked in the Bandit class)
        avg_regret = sum(self.regrets) / len(self.regrets)
        logger.info(f'Average regret for {bandit_name} using {algorithm}: {avg_regret}')


# In[37]:


class Visualization():

    def plot1(self, bandit_names, rewards):
        # Visualize the performance of each bandit: linear and log
        plt.figure(figsize=(10, 5))

        # Linear plot
        plt.subplot(1, 2, 1)
        for i in range(len(bandit_names)):
            plt.plot(rewards[i], label=bandit_names[i])
        plt.xlabel('Trials')
        plt.ylabel('Reward')
        plt.title('Reward vs Trials (Linear)')
        plt.legend()

        # Log plot
        plt.subplot(1, 2, 2)
        for i in range(len(bandit_names)):
            plt.plot(rewards[i], label=bandit_names[i])
        plt.xlabel('Trials')
        plt.ylabel('Reward')
        plt.title('Reward vs Trials (Log)')
        plt.legend()
        plt.yscale('log')

        plt.tight_layout()
        plt.show()

    def plot2(self, cumulative_rewards_egreedy, cumulative_rewards_thompson, cumulative_regrets_egreedy, cumulative_regrets_thompson):
        # Compare E-greedy and thompson sampling cumulative rewards and regrets
        plt.figure(figsize=(10, 5))

        # Cumulative rewards comparison
        plt.subplot(1, 2, 1)
        plt.plot(cumulative_rewards_egreedy, label='E-Greedy')
        plt.plot(cumulative_rewards_thompson, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Rewards')
        plt.title('Cumulative Rewards Comparison')
        plt.legend()

        # Cumulative regrets comparison
        plt.subplot(1, 2, 2)
        plt.plot(cumulative_regrets_egreedy, label='E-Greedy')
        plt.plot(cumulative_regrets_thompson, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Regrets')
        plt.title('Cumulative Regrets Comparison')
        plt.legend()

        plt.tight_layout()
        plt.show()


# In[73]:


class EpsilonGreedy(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.epsilon = 1.0

    def __repr__(self):
        return f'A Actual reward: {self.reward}, estimated reward {self.reward_estimate}'

    def pull(self):
        return np.random.normal(self.reward, 1)

    def update(self, x):
        self.N += 1    # Increase count of iterations by 1
        self.reward_estimate = ((self.N - 1)*self.reward_estimate + x) / self.N    # Update the reward estimate
        self.learning_process.append(self.reward_estimate)    # Add the updated value to the reward list

    def experiment(self, num_trials):
        # Running an experiment with num_trials iterations
        for _ in range(num_trials):
            arm = self.pull()
            reward = Bandit_Reward[arm]
            self.rewards.append(reward)  # Recording the reward
            regret = max(Bandit_Reward) - reward  # Calculating regret
            self.regrets.append(regret)  # Recording regret
            self.update(arm, reward)  # Updating bandit's state

    def report(self):
        super().report("Epsilon-Greedy Bandit", self.rewards, "Epsilon-Greedy")


# In[74]:


class ThompsonSampling(Bandit):
    def __init__(self, p, precision):
        super().__init__(p)
        self.precision = precision
        self.m = 0              # The estimated reward
        self.lambda_ = 1        # Parameter lambda
        self.tau = 1            # Parameter tau
        self.N = 0              # Iteration count
        self.sum_x = 0          # Cumulative reward

    def __repr__(self):
        return f'A Actual reward: {self.reward}, estimated reward {self.m}'

    def pull(self):
        # Sample from Beta distribution for each arm
        return np.random.randn() / np.sqrt(self.tau) + self.reward

    def update(self, chosen_arm, reward):
        self.lambda_ += self.tau    # Update lambda
        self.sum_x += x                     # Update the cumulative sum
        self.m = (self.tau * self.sum_x) / self.lambda_    # Update the reward estimate
        self.learning_process.append(self.m)    # Update learning process list
        self.N += 1                        # Increment iteration count

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            chosen_arm = self.pull()
            reward = np.random.choice(self.p[chosen_arm])
            rewards.append(reward)
            self.rewards.append(reward)  # Collect rewards
            self.update(chosen_arm, reward)
        return rewards

    def report(self):
        super().report("Thompson Sampling Bandit", self.rewards, "Thompson Sampling")


# In[75]:


def comparison(epsilon_greedy_rewards, thompson_rewards, epsilon_greedy_regrets, thompson_regrets):
    # Plot cumulative rewards and cumulative regrets for both algorithms
    plt.figure(figsize=(10, 5))

    # Cumulative rewards comparison
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(epsilon_greedy_rewards), label='Epsilon-Greedy')
    plt.plot(np.cumsum(thompson_rewards), label='Thompson Sampling')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards Comparison')
    plt.legend()

    # Cumulative regrets comparison
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(epsilon_greedy_regrets), label='Epsilon-Greedy')
    plt.plot(np.cumsum(thompson_regrets), label='Thompson Sampling')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Regrets')
    plt.title('Cumulative Regrets Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[76]:


Bandit_Reward=[1,2,3,4]
NumberOfTrials: 20000
epsilon_greedy_bandit.epsilon = 0.1

