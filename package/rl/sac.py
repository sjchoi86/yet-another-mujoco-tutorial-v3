import collections,random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

def list2torch(x,device):
    """
        List to torch
    """
    return torch.tensor(np.array(x),dtype=torch.float).to(device)

# Replay buffer
class ReplayBufferClass():
    """
        Replay buffer
    """
    def __init__(self,buffer_limit=100000,device='cpu') -> None:
        """
            Initialize buffer
        """
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def size(self):
        """
            Get size
        """
        return len(self.buffer)

    def clear(self):
        """
            Clear
        """
        self.buffer.clear()

    def put(self,item):
        """
            Put
        """
        self.buffer.append(item)
    
    def put_mini_batch(self,mini_batch):
        """
            Batch put
        """
        for transition in mini_batch:
            self.put(transition)
        
    def sample(self,n):
        """
            Sample
        """
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append(a)
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done_mask else 1.0
            done_mask_list.append([done_mask])
        return list2torch(s_list,self.device),list2torch(a_list,self.device),list2torch(r_list,self.device),\
            list2torch(s_prime_list,self.device),list2torch(done_mask_list,self.device)

# SAC implementations are based on the following paper
"""
    https://arxiv.org/pdf/1812.05905.pdf
"""    
    
# Actor
class ActorClass(nn.Module):
    def __init__(self,
                 name       = "actor",
                 obs_dim    = 8,
                 h_dims     = [256,256],
                 out_dim    = 1,
                 max_out    = 1.0,
                 init_alpha = 0.1,
                 lr_actor   = 0.0003,
                 lr_alpha   = 0.0003,
                 device     = None) -> None:
        super(ActorClass, self).__init__()
        # Initialize
        self.name       = name
        self.obs_dim    = obs_dim
        self.h_dims     = h_dims
        self.out_dim    = out_dim
        self.max_out    = max_out
        self.init_alpha = init_alpha
        self.lr_actor   = lr_actor
        self.lr_alpha   = lr_alpha
        self.device     = device
        self.init_layers()
        self.init_params()
        # Set optimizer
        self.actor_optimizer = optim.Adam(self.parameters(), lr=self.lr_actor)
        self.log_alpha = torch.tensor(np.log(self.init_alpha),requires_grad=True,dtype=torch.float32,device=self.device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        h_dim_prev = self.obs_dim
        for h_idx, h_dim in enumerate(self.h_dims):
            self.layers['mlp_{}'.format(h_idx)] = nn.Linear(h_dim_prev, h_dim)
            self.layers['relu_{}'.format(h_idx)] = nn.ReLU()
            h_dim_prev = h_dim
        self.layers['mu'] = nn.Linear(h_dim_prev, self.out_dim)
        self.layers['std'] = nn.Linear(h_dim_prev, self.out_dim)

        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.parameters = nn.ParameterDict(self.param_dict)

    def init_params(self):
        """
            Initialize parameters
        """
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x,SAMPLE_ACTION=True):
        """
            Forward
        """
        x = x.to(self.device)
        for h_idx, _ in enumerate(self.h_dims):
            x = self.layers['relu_{}'.format(h_idx)](self.layers['mlp_{}'.format(h_idx)](x))
        # Compute mean and std
        mean = self.layers['mu'](x)
        # std  = F.softplus(self.layers['std'](x)) + 1e-6
        std = torch.sigmoid(self.layers['std'](x))
        # Define Gaussian
        GaussianDistribution = Normal(mean, std)
        # Sample action
        if SAMPLE_ACTION:
            action = GaussianDistribution.rsample()
        else:
            action = mean
        # Compute log prob
        log_prob = GaussianDistribution.log_prob(action)
        # Normalize action
        real_action = torch.tanh(action) * self.max_out
        real_log_prob = log_prob - torch.log(self.max_out*(1-torch.tanh(action).pow(2)) + 1e-6)
        return real_action, real_log_prob
    
    def train(self,
              q_1,
              q_2,
              target_entropy,
              mini_batch):
        """
            Train
        """
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob 
        
        q_1_value = q_1(s, a)
        q_2_value = q_2(s, a)
        q_1_q_2_value = torch.cat([q_1_value, q_2_value], dim=1)
        min_q_value   = torch.min(q_1_q_2_value, 1, keepdim=True)[0]
        
        # Update actor
        actor_loss = -min_q_value -entropy # Eq. (7)
        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()
        
        # Automating Entropy Adjustment
        alpha_loss = -(self.log_alpha.exp() * (log_prob+target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
# Critic
class CriticClass(nn.Module):
    def __init__(self,
                 name      = "critic",
                 obs_dim   = 75,
                 a_dim     = 8,
                 h_dims    = [256,256],
                 out_dim   = 1,
                 lr_critic = 0.0003,
                 device    = None) -> None:
        """
            Initialize Critic
        """
        super(CriticClass, self).__init__()
        # Initialize
        self.name      = name
        self.obs_dim   = obs_dim
        self.a_dim     = a_dim
        self.h_dims    = h_dims
        self.out_dim   = out_dim
        self.lr_critic = lr_critic
        self.device    = device
        self.init_layers()
        self.init_params()
        # Set optimizer
        self.critic_optimizer = optim.Adam(self.parameters(),lr=self.lr_critic)

    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        h_dim_prev = self.h_dims[0]
        for h_idx, h_dim in enumerate(self.h_dims):
            if h_idx == 0:
                self.layers['obs']      = nn.Linear(self.obs_dim, int(self.h_dims[0]/2))
                self.layers['obs_relu'] = nn.ReLU()
                self.layers['act']      = nn.Linear(self.a_dim, int(self.h_dims[0]/2))
                self.layers['act_relu'] = nn.ReLU()
            else:
                self.layers['mlp_{}'.format(h_idx)] = nn.Linear(h_dim_prev, h_dim)
                self.layers['relu_{}'.format(h_idx)] = nn.ReLU()
            h_dim_prev = h_dim
        self.layers['out'] = nn.Linear(h_dim_prev, self.out_dim)

        # Accumulate layers weights
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.parameters = nn.ParameterDict(self.param_dict)

    def init_params(self):
        """
            Initialize parameters
        """
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self,
                x,
                a):
        x = x.to(self.device)
        a = a.to(self.device)
        for h_idx, _ in enumerate(self.h_dims):
            if h_idx == 0:
                x = self.layers['obs_relu'](self.layers['obs'](x))
                a = self.layers['act_relu'](self.layers['act'](a))
                cat = torch.cat([x,a], dim=1)
            else:
                 q = self.layers['relu_{}'.format(h_idx)](self.layers['mlp_{}'.format(h_idx)](cat))
        q = self.layers['out'](q)
        return q
    
    def train(self,
              target,
              mini_batch):
        """
            Train
        """
        s, a, r, s_prime, done = mini_batch
        critic_loss = F.smooth_l1_loss(self.forward(s,a), target)
        self.critic_optimizer.zero_grad()
        critic_loss.mean().backward()
        self.critic_optimizer.step()
        
    def soft_update(self, tau, net_target):
        """
            Soft update of Critic
        """
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            
# Bellman backup operator
def get_target(pi, q1, q2, gamma, mini_batch, device):
    q1 = q1.to(device)
    q2 = q2.to(device)
    pi = pi.to(device)
    s, a, r, s_prime, done = mini_batch
    with torch.no_grad():
        # We use the action the current 'pi' making SAC off-policy
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob # incentivize exploration
        
        # Use the minimum Q among q1 and q2 (to avoid over-confident issue)
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q, 1, keepdim=True)[0]
        
        # Get target using Bellman backup operator
        target = r + gamma * done * (min_q + entropy.mean()) # Eg. (3)
    return target 
