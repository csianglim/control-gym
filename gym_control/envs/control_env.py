# Import gym libraries
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# Import dependencies
import numpy as np
import control
import random
import math

# CONSTANTS
SAMPLING_TIME = 1
NUM = [0.05,0]  	# Transfer function NUM
DEN = [1,-0.6]  	# Transfer function DEN
U_MIN = -50 		# Minimum steam flow rate U_MIN
U_MAX = 50			# Maximum steam flow rate U_MAX
c = 10 				# Reward onstant defined in the ADCONIP paper
epsilon = 0.1 		# Error threshold for setpoint tracking
n_steps = 5 		# How many steps below threshold before an episode is done
magicNumber = 10 	# WHAT IS THIS
setpoint = 2 		# setpoint
maxSteps = 200

# Control Gym Environment
class ControlEnv(gym.Env):

	## Required methods for gym.Env
	def __init__(self):
		self.sys = control.matlab.tf(NUM,DEN,SAMPLING_TIME)
		self.u = []
		self.t = []
		self.action_space = spaces.Box(low=U_MIN, high=U_MAX, shape=(1,))
		self.observation_space = spaces.Box(low=U_MIN, high=U_MAX, shape=(1,))

	def _step(self, action):
		self.action = np.clip(action, U_MIN, U_MAX)
		assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

		# Step forward in time by 1 second
		self.u = np.append(self.u,action)
		self.t = np.append(self.t,len(self.t))
		self.y_out,_,_ = control.matlab.lsim(self.sys,self.u,self.t)
		
		# The last time point
		self.y_t = np.asscalar(self.y_out[:,-1])
		self.error = abs(setpoint - self.y_t)   
		
		# Calculate reward for this step
		if abs(self.y_t - setpoint) < epsilon:        
			self.reward = 10
		else:    
			self.reward = -abs(self.y_t - setpoint)

		# Check if this episode is done
		done = False
		if np.asscalar(self.t[-1]) >= (maxSteps + magicNumber):
			done = True
            
		if self.error < epsilon:
			self.tracked_steps += 1        
			if self.tracked_steps >= n_steps:
				done = True
 	  
 	  # Return tuple of obs, reward, done, metadata
		return self.y_t, self.reward, done, {}

	def _reset(self):
		# Initialize arrays
		self.u = []
		self.t = []

		# The control action, u, is the steam flow rate
		# Initialize with 1D column vector of random control action
		self.u = random.uniform(U_MIN,U_MAX)*np.ones([magicNumber,1])

		# We will simulate a bunch of time points but only grab the last one as the return value
		n = len(self.u) # Number of time points
		self.t = np.array([np.linspace(0,n-1,n)]).T # 1D column vector of time points	

		# lsim returns the output given the transfer function, action and time points
		self.y_out,_,_ = control.matlab.lsim(self.sys,self.u,self.t) 
		self.tracked_steps = 0

		# The system output, y, is the current moisture content
		# Grab the last value in the array and return a scalar
		return np.asscalar(self.y_out[:,-1])

	def render(self, mode='human', close=False):
		pass