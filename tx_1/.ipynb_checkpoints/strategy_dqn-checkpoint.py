import numpy as np
import tensorflow as tf
import os
from collections import deque
import random
import time


use_jit = False

class DQNAgent:
    def __init__(self, model,
                 strategy,
                 target_model,
                 n_actions,
                 memory_size = 100000, 
                 optimizer = tf.keras.optimizers.Adam(0.0005), 
                 gamma = 0.99,
                 batch_size =32,
                 name = "dqn1",
                 target_model_sync = 1000,
                 exploration = 0.01,
                ):
        self.strategy = strategy
        self.num_replicas = strategy.num_replicas_in_sync
        self.exploration = exploration
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.model = model
        self.name = name
        self.memory_size = memory_size
        self.optimizer = optimizer
        self.m1 = np.eye(self.n_actions, dtype="float32")
        self.target_model = target_model #tf.keras.models.clone_model(self.model)
        self.target_model_sync = target_model_sync
        self.num_model_inputs = len(self.model.inputs)
        self.num_envs = 0
        if not os.path.exists("logs"):
            os.mkdir("logs")
            print("created ./logs")
        self.memory = deque(maxlen = self.memory_size)
      
    
    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights())
      
    def load_weights(self):
        self.model.load_weights(self.name+".h5")
    def save_weights(self):
        self.model.save_weights(self.name+".h5", overwrite = True)
        
    @tf.function(jit_compile = use_jit)
    #@tf.function()
    def model_call(self, x):
        return tf.math.argmax(self.model(x), axis = 1)
    
    def select_actions(self, states_array):

        if random.random() < self.exploration:
            return tf.random.uniform(shape=[self.num_envs], minval=0, maxval=self.n_actions, dtype=tf.int32).numpy()

        assert self.num_envs % self.strategy.num_replicas_in_sync == 0
        inc = int(self.num_envs/self.strategy.num_replicas_in_sync)

        self.tn = -inc
        def vfunc(v):
            self.tn+=inc
            values = [states_array[o][self.tn:self.tn+inc] for o in range(self.num_model_inputs)]
            return values

        inp = (self.strategy.experimental_distribute_values_from_function(vfunc))
        ret = self.strategy.run(self.model_call, args = (inp,))
        
        ret = np.array([x.numpy() for x in ret.values]).reshape(-1)
        return ret


        
    def observe_sasrt(self, state, action, next_state, reward, terminal):
        self.memory.append([state, action, reward, 1-int(terminal), next_state])
        
    @tf.function(jit_compile = use_jit)
    #@tf.function()
    def get_target_q(self, next_states, rewards, terminals):
        estimated_q_values_next = self.target_model(next_states)
        q_batch = tf.math.reduce_max(estimated_q_values_next, axis=1)
        target_q_values = q_batch * self.gamma * terminals + rewards
        return target_q_values

        
    @tf.function(jit_compile = use_jit)
    #@tf.function()
    def tstep(self, data):
        states, next_states, rewards, terminals, masks = data
        target_q_values = self.get_target_q(next_states, rewards, terminals)
        
        with tf.GradientTape() as t:
            model_return = self.model(states, training=True) 
            mask_return = model_return * masks
            estimated_q_values = tf.math.reduce_sum(mask_return, axis=1)
            #print(estimated_q_values, mask_return, model_return, masks)
            loss_e = tf.math.square(target_q_values - estimated_q_values)
            loss = tf.reduce_mean(loss_e)
        
        
        gradient = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        
        return loss, tf.reduce_mean(estimated_q_values)
    
    
    def data_get_func(self, _n):
        idx = np.random.randint(0, len(self.memory), self.batch_size)
        sarts_batch = [self.memory[i] for i in idx]
        
        states = [x[0] for x in sarts_batch]
        states_array = []
        for i in range(self.num_model_inputs):
                states_array.append(np.array([x[i] for x in states], dtype = "float32"))
        
                    
        actions = [x[1] for x in sarts_batch]
        rewards = np.array([x[2] for x in sarts_batch], dtype="float32")
        terminals = np.array([x[3] for x in sarts_batch], dtype="float32")
        
        next_states = [x[4] for x in sarts_batch]
        next_states_array = []
        for i in range(self.num_model_inputs):
                next_states_array.append(np.array([x[i] for x in next_states], dtype = "float32"))
                
        
        #print(actions)
        masks = np.array(self.m1[actions])
        return [states_array, next_states_array, rewards, terminals, masks]

    def update_parameters(self):
        self.total_steps_trained+=1
        if self.total_steps_trained % self.target_model_sync == 0:
            self.copy_weights()

           

        distributed_values = (self.strategy.experimental_distribute_values_from_function(self.data_get_func))
        return  self.strategy.reduce(tf.distribute.ReduceOp.MEAN, self.strategy.run(self.tstep, args = (distributed_values,)), axis = None)
   
    
    def train(self, num_steps, envs, log_interval = 1000, warmup = 0):
        self.total_steps_trained = -1

        num_envs = len(envs)
        self.num_envs = num_envs
        states = [x.reset(True) for x in envs]
        
        times= deque(maxlen=10)
        start_time = time.time()
        
        self.rewards = [0]
        self.losses = [0]
        self.q_v = [0]
        
        def save_current_run():
            self.save_weights()
            if len(self.losses) > 0:
                file = open("logs/loss_log.txt", "a")  
                file.write(str(np.mean(self.losses)))
                file.write("\n")
                file.close()
            if len(self.q_v) > 0:
                file = open("logs/qv_log.txt", "a")  
                file.write(str(np.mean(self.q_v)))
                file.write("\n")
                file.close()

            file = open("logs/rewards_log.txt", "a")  
            file.write(str(np.mean(self.rewards)))
            file.write("\n")
            file.close()
            
    

            self.rewards = []
            self.losses = []
            self.q_v = []
        
        try:
            for i in range(num_steps):
                if i % log_interval == 0:
                    progbar = tf.keras.utils.Progbar(log_interval, interval=0.1, stateful_metrics = ["t", "rewards"])

                states_array = []
                for o in range(self.num_model_inputs):
                        states_array.append(np.array([x[o] for x in states], dtype = "float32"))
                
                
                actions = self.select_actions(states_array)
                
                sasrt_pairs = []
                for index in range(num_envs):
                    sasrt_pairs.append([states[index], actions[index]]+[x for x in envs[index].step(actions[index])])

                next_states = [x[2] for x in sasrt_pairs]

                reward = [x[3] for x in sasrt_pairs]
                
                
                self.rewards.extend(reward)
                    
                for index, o in enumerate(sasrt_pairs):
                    #print(o)
                    if o[4] == True:
                        next_states[index] = envs[index].reset()
                    self.observe_sasrt(o[0], o[1], o[2], o[3], o[4])

                states = next_states
                
                if i > warmup:
                        loss, q = self.update_parameters()
                        self.losses.append(loss.numpy())
                        self.q_v.append(q.numpy())
                        

                else:
                    loss, q = 0, 0

                end_time = time.time()
                elapsed = (end_time - start_time) * 1000
                times.append(elapsed)
                start_time = end_time



                progbar.update(i%log_interval+1, values = 
                               [("loss", np.mean(self.losses[-1]) if len(self.losses)>0 else 0),
                                ("mean q", np.mean(self.q_v[-1]) if len(self.q_v)>0 else 0),
                                ("rewards", np.mean(self.rewards) if len(self.rewards)>0 else 0),
                                ("t", np.mean(times))], 
                              finalize = (i+1) % log_interval == 0)
        
                if (i+1) % log_interval == 0:
                    save_current_run()
                    
                    
        except KeyboardInterrupt:
            print("\n\nbreak!")
        
        save_current_run()
   