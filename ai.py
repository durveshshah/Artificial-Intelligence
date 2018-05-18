import numpy as np #nump is for arrays
import random # Experience Replay
import os # Load the model from the brain
import torch # Used for Nueral Network
import torch.nn as nn #some tools for Nueral Networks 
import torch.nn.functional as F # functions for implementing Nueral Network
import torch.optim as optim # Used for optimizing and for stochastic gradient decent
import torch.autograd as autograd #puting Tensors(Advanced arrays) into Variable
from torch.autograd import Variable # Same as above 

#Creating architecture of Neural Network

class Network(nn.Module): #Inheritence of a class

      def __init__(self, input_size, nb_action): #self variable is attached to the object, and there is input and output is nb_action
              super(Network,self).__init__()
          self.input_size = input_size  #number of inputs
          self.nb_action = nb_action  #number of outputs
          self.fcl = nn.Linear(input_size, 100) #connections from input layer to 30 hidden layers
          self.fc2 = nn.Linear(100, nb_action) #connections from hidden layers to output layers
      
      def forward(self, state): #input state for going left,right or straight
          x = F.relu(self.fc1(state)) # activating hidden layers i.e x and applying rectifier function
          q_values = self.fc2(x) # activating from hidden layers to output
          return q_values # return output values wheather you want to go straight, left or right
      
#Implementing Experience Replay
class ReplayMemory(object):
          
      def __init__(self,capacity): #capacity for example: 100 last events
          self.capacity = capacity
          self.memory = [] #initialize memory
    
      def push(self, event): # append push function into the memory
           self.memory.append(event) #event will append last state, new state, last action and last reward to the memory
           if len(self.memory) > self.capacity: # if this is true then delete the old events(first events are the oldest one) and make the capacity equal to the memory
               del self.memory[0]
               
      def sample(self, batch_size):
          # if list = ((1,2,3), (4,5,6)), then zip(*list) = ((1,4), (2,5), (3,6))
          # zip reshapes the format as shown above (1,4) is state1 state2, (2,5) is action1 and action2 (3,6) is reward 1 and reward2
          
          samples = zip(*random.sample(self.memory, batch_size)) 
          # map function is used to import samples into pytorch variable
          return map(lambda x: Variable(torch.cat(x,0))) # Used to  conactenate lambda x to its first dimension and converts Variable into torch function that will contain torch and gradient
          
#Implementing Deep Q Learning
          
class Dqn():
          
       def __init__(self, input_size, nb_action, gamma):
           self.gamma = gamma
           self.reward_window = [] # mean of the last 100 rewards
           self.model = Network( input_size, nb_action)
           self.memory = ReplayMemory(100000) # Called ReplayMemory class for 1 lakh transitions
           self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001) # Add self.model into it and the learning rate. The larger the learning rate the better functioning of self driving car
           self.last_state = torch.Tensor(input_size).unsqueeze(0) # last_state is the input variable. Tensor is basically an array which takes one single type
           # we must include fake dimension so that tensor can take it in batches. The fake dimension is the first dimension of the last_state.
           # Index in python start from zero. zero is the fake dimension
           self.last_action = 0  # action can be 0,1 or 2 which represents action to angle rotation [0,20,-20] respectievely.
           self.last_reward = 0 # reward is  float number which is betwen -1 and +1
           
      def select_action(self,state): # Actions are inputs. Input of the neural networks are the input states
          probs = F.softmax(self.model(Variable(state, volatile = True))*200) # T=7 probs calculates Q-values q1,q2 and q3.
          # volatile equals true means it will include gradient
          # T is Temperature parameter which decides the direction of the car it should play for example,
          # softmax([1,2,3]) = [0.04,0.11,0.85] >= softmax([1,2,3]*3) = [0,0.02,0.98]. Here the greater the temperature value i.e T the higher the chances of car to take right direction
          # Here, it will play 0.98 action because the probability is higher
          action = probs.multinomial() # Multinomial function will draw random distribution from softmax
          return action.data[0,0] # we have to return out action with respect to 1st dimension i.e the fake dimension
           
      
     def learn(self,batch_state, batch_next_state, batch_reward, batch_action):
         outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) # gather will automatically chooese best action to play i.e where to go and 1 is we only want action which we have choosen
         # self.model is the output and batch_state are the inputs
         # we have to do unsqueeze again because of the action the car wants to take and squeeze to kill the fake dimension 
         next_outputs = self.model(batch_next_state).detach().max(1)[0] # Q values of the next state is represented by index 0 and max 1 is the action
         # detach is detaching all the values and calculating the maximum one
         target = self.gamma*next_outputs + batch_reward
         td_loss = F.smooth_l1_loss(outputs, target)
         self.optimizer.zero_grad() # zero_grad will reinitialize the optimizer at each iteration of the loop
         
         
         # Performing back propogation
         td_loss.backward(retain_variables = True) # tThis back propogates and retain variables = true is used to free the memory 
         self.optimizer.step() # step fucntion will update the weights
          
     def update(self, reward,new_signal): # the reward and signal is the last reward in the map.py. Based on this it will take further action
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])) # Every state has been assigned a tensor so we have to assign it on a last_action also
        # push method is called to update the memory
        
         action = self.select_action(new_state) # Play new action after reaching new state
         if(len.self.memory.memory.memory) > 100: # second .memory is the attribute on line 32 i.e self.memory[] in replay object method
             batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
             self.learn(batch_state, batch_next_state, batch_reward, batch_action)
         self.last_action = action # updating
         self.last_state = new_state # updating
         self.last_reward = reward # updating
         self.reward_window.append(reward) # updating reward window i.e mean of rewards
         if len(self.reward_window) > 1000: # calculate 1000 means of last 100 rewards
             del self.reward_window[0]
         return action 
     
     def score(self):
         return sum(self.reward_window)/(len(self.reward_window)+1) # Calculate mean i. sum/number of elements in reward window
         # reward_window must not be equal to 0 thats why we have added +1. if it is 0 the system will crash
     def sace(self):
             torch.save({'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict,},
                         'last_brain.pth') # state_dict is the key and self.model is the neural network which is the object. Similarly for the optimizer
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])  # key is state_dict
            self.optimizer.load_state_dict(checkpoint['optimizer']) # key is optimizer 
            print("done !")
            
            else:
                print("No checkpoint found...")
        
             
