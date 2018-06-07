#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn

seaborn.set()

random.seed(42)
np.random.seed(42)

# all states
N_STATES = 19

# all states but terminal states
states = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state values from Bellman equation
realStateValues = np.arange(-20, 22, 2) / 20.0
realStateValues[0] = realStateValues[N_STATES + 1] = 0.0

# base class for lambda-based algorithms in this chapter
# In this example, we use the simplest linear feature function, state
# aggregation. And we use exact 19 groups, so the weights for each group is
# exact the value for that state
class ValueFunction:
    # @rate: lambda, as it's a keyword in python, so I call it rate
    # @stepSize: alpha, step size for update
    def __init__(self, rate, stepSize):
        self.rate = rate
        self.stepSize = stepSize
        self.weights = np.zeros(N_STATES + 2)

    # the state value is just the weight
    def value(self, state):
        return self.weights[state]

    # feed the algorithm with new observation
    # derived class should override this function
    def learn(self, state, reward):
        return

    # initialize some variables at the beginning of each episode
    # must be called at the very beginning of each episode
    # derived class should override this function
    def newEpisode(self):
        return

# TD(lambda) algorithm
class TemporalDifferenceLambda(ValueFunction):
    def __init__(self, rate, stepSize):
        ValueFunction.__init__(self, rate, stepSize)
        self.newEpisode()

    def newEpisode(self):
        # initialize the eligibility trace
        self.eligibility = np.zeros(N_STATES + 2)
        # initialize the beginning state
        self.lastState = START_STATE

    def learn(self, state, reward):
        # update the eligibility trace and weights
        self.eligibility *= self.rate
        self.eligibility[self.lastState] += 1
        delta = reward + self.value(state) - self.value(self.lastState)
        delta *= self.stepSize
        self.weights += delta * self.eligibility
        self.lastState = state

# 19-state random walk
def randomWalk(valueFunction):
    valueFunction.newEpisode()
    currentState = START_STATE
    while currentState not in END_STATES:
        newState = currentState + np.random.choice([-1, 1])
        if newState == 0:
            reward = -1
        elif newState == N_STATES + 1:
            reward = 1
        else:
            reward = 0
        valueFunction.learn(newState, reward)
        currentState = newState

def figure_(valueFunctionGenerator, runs, lambdas, alphas):
    # play for 10 episodes for each run
    episodes = 100
    # track the rms errors
    errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
    for run in range(runs):
        for lambdaIndex, rate in zip(range(len(lambdas)), lambdas):
            for alphaIndex, alpha in zip(range(len(alphas[lambdaIndex])),
                                         alphas[lambdaIndex]):
                valueFunction = valueFunctionGenerator(rate, alpha)
                for episode in range(episodes):
                    print('run:', run, 'lambda:', rate, 'alpha:', alpha,
                          'episode:', episode)
                    randomWalk(valueFunction)
                    stateValues = [valueFunction.value(state)
                                   for state in states]
                    errors[lambdaIndex][alphaIndex] += np.sqrt(np.mean(
                        np.power(stateValues - realStateValues[1: -1], 2)))

    # average over runs and episodes
    for error in errors:
        error /= episodes * runs
    plt.figure()
    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], label='lambda = ' + str(lambdas[i]))
    plt.xlabel('learning rate')
    plt.ylabel('RMSE')

# Figure: TD(lambda) algorithm
def td_figure():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.99, 0.09),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.33, 0.03),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01),
              np.arange(0, 0.044, 0.004)]
    figure_(TemporalDifferenceLambda, 1, lambdas, alphas)


td_figure()
plt.gca().set_ylim((0.0, 0.6))
plt.savefig('td_lambda_random_walk.eps')
