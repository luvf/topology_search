BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 128
DEVICE = "cpu"
# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

# Get number of actions from gym action space
import torch
from torch import nn

from collections import namedtuple
import itertools

from torch.functional import F


from pypownet.agent import Agent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class Encoder(object):
    """the tool of sklearn didn't work for me
        create a dict tha associate actions to numbers

    """
    def __init__(self):
        self.keys = dict()
        self.values = list()

    def fit(self,data):
        for x in data:
            if not tuple(x) in self.keys.keys():
                self.keys[tuple(x)]= len(self.values)
                self.values.append(tuple(x))
        return self

    def encode(self, data):
        return [self.keys[tuple(d)] for d in data] 

    def decode(self, data):
        return [self.values[int(d)] for d in data]






class NNModel(nn.Module):
    """docstring for Generator"""
    def __init__(self,input_size, output_size):
        super(NNModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.clf = nn.Sequential(
            #nn.Linear(64 * 6 * 6, 1024),
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace= True),
            nn.Linear(1024, output_size),
            #nn.Softmax(),
        )
        

    def forward(self, x):
        #x = x.view(x.size(0), 64 * 6 * 6)
        x = self.clf(x)
        return x






class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args) #Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def action_space(environment):
    """ lists all unary actions"""
    actions = list()
    ap = environment.action_space
    actions.append(torch.tensor(ap.get_do_nothing_action(),dtype=torch.float))

    number_lines = ap.lines_status_subaction_length
    for l in range(number_lines):
               
        action = ap.get_do_nothing_action(as_class_Action=True)
        ap.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
        actions.append(torch.tensor(action.as_array(), dtype=torch.float))


        # For every substation with at least 4 elements, try every possible configuration for the switches
    for substation_id in ap.substations_ids:
        substation_n_elements = ap.get_number_elements_of_substation(substation_id)
        #if 6 > substation_n_elements > 3:
        if substation_n_elements > 3:
            # Look through all configurations of n_elements binary vector with first value fixed to 0
            for configuration in list(itertools.product([0, 1], repeat=substation_n_elements - 1)):
                new_configuration = [0] + list(configuration)
                action = ap.get_do_nothing_action(as_class_Action=True)
                ap.set_substation_switches_in_action(action=action, substation_id=substation_id,
                                                               new_values=new_configuration)
                actions.append(torch.tensor(action.as_array(),dtype=torch.float))
    return actions



import random
from torch import optim
import math
class DQNAgent(Agent):
    """ 

    """

    def __init__(self, environment):
        """Initialize a new agent."""
        
        self.environment = environment 
        self.Train = True
        self.steps = 0


        self.actions = action_space(environment)
        #obs_dim = environment.observation_space.as_array()
        obs_shape = environment.observation_space.shape
        input_dim = sum(list(zip(*obs_shape))[0])+ environment.action_space.get_do_nothing_action().shape[0]


        self.policy_net = NNModel(input_dim, 1).to(DEVICE)
        self.target_net = NNModel(input_dim, 1).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(1000)
        self.current_obs = torch.zeros(sum(list(zip(*obs_shape))[0]))
        self.load_model()


    def best_value(self, obs, net):
        inputs = [torch.cat((obs, a)).view(1,-1) for a in self.actions]
        predictions = net(torch.cat(inputs))
        return torch.max(predictions)

    def best_action(self, obs, net):
        inputs = [torch.cat((obs, a)).view(1,-1) for a in self.actions]
        predictions = net(torch.cat(inputs))
        return torch.argmax(predictions)


        #f#or a in self.actions:

    def act(self, observation):
        obs= torch.tensor(observation, dtype= torch.float)
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps / EPS_DECAY)
        sample = random.random()

        self.steps +=1

        if self.Train == False:
            eps_threshold = 0#on explore pas
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                prediction = self.best_action(obs, self.policy_net)
                return self.actions[prediction]

        else:
            return self.actions[random.randrange(len(self.actions))]

 

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.optimizer.zero_grad()
        self.policy_net.train()
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                      batch.next_state)), device=DEVICE, dtype=torch.uint8)
        
     
        #print([x.view(1,-1) for x in batch.state])
        state_batch = torch.cat([x.view(1,-1) for x in batch.state])#freakin nasty
        action_batch = torch.cat([x.view(1,-1) for x in batch.action])


        state_action_values = self.policy_net(torch.cat((state_batch[0],action_batch[0])))

        
        
        
        
        # Compute the expected Q values
        reward_batch = torch.cat([x.view(1,-1) for x in batch.reward])
        next_state_values = torch.cat([self.best_value(s, self.target_net).view(-1,1) for s in batch.next_state])

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        
        loss.backward()
        #for param in policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.policy_net.eval()

    def to_action(self, prediction):
        return self.actions[torch.argmax(prediction)]


    def feed_reward(self, action, observation, rewards_aslist):
        self.last_obs = self.current_obs
        self.current_obs = torch.tensor(observation,device= DEVICE, dtype=torch.float )
        
        action = torch.tensor(action, dtype= torch.float)

        self.memory.push(self.last_obs,action, self.current_obs, torch.tensor(sum(rewards_aslist),device= DEVICE, dtype= torch.float))
        if self.steps % (8) == 0 :
            self.optimize_model()


        if self.steps % TARGET_UPDATE == 0:
            
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "ml/DQN.save")


    def load_model(self):
        self.policy_net.load_state_dict(torch.load("ml/DQN.save"))


    def __del__(self):
        self.save_model()


