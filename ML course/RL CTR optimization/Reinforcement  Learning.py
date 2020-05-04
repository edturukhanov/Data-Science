import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Upper Confidence Bound

#Data import
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
ads_selected = []
number_of_selection = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (number_of_selection[i] > 0):
            average_reward = sums_of_rewards[i]/number_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selection[ad] = number_of_selection[ad] + 1
    reward = data.values[n, ad]
    sums_of_rewards[ad] =  sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    

#Visualising the result
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times selected')
plt.show()

#====================================================
#Thompson Sampling
import random

#Data import
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = data.values[n, ad]
    if reward ==1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    

#Visualising the result
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times selected')
plt.show()

