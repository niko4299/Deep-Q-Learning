import numpy as np
import torch as T
from dueling_deep_q_network import DuelingDeepQNetwork
from basic_agent import Agent

class DuelingDDQNAgent(Agent):
  def __init__(self,*args,**kwargs):
    super(DuelingDDQNAgent,self).__init__(*args, **kwargs)
    self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_eval',
                                chkpt_dir=self.chkpt_dir)

    self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_next',
                                chkpt_dir=self.chkpt_dir)

  def choose_action(self, observation):
    if np.random.random() > self.epsilon:
        state = T.tensor([observation], dtype=T.float).to(
            self.q_eval.device)
        _, advantage = self.q_eval.forward(state)
        action = T.argmax(advantage).item()
    else:
        action = np.random.choice(self.action_space)

    return action


  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
        return
    self.q_eval.optimizer.zero_grad()
    self.replace_target_network()
    states, actions, rewards, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)
    V_s, A_s = self.q_eval.forward(states)
    V_s_, A_s_ = self.q_next.forward(states_)
    V_s_eval, A_s_eval = self.q_eval.forward(states_)
    
    q_pred = T.add(V_s, (A_s - A_s.mean(dim = 1, keepdim = True)))[indices, actions]
    q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim = 1, keepdim = True)))
    q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim = 1, keepdim = True)))

    max_actions = T.argmax(q_eval,dim = 1)

    q_next[dones] = 0.0

    q_target = rewards + self.gamma * q_next[indices, max_actions]
    
    loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
    loss.backward()
    self.q_eval.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()