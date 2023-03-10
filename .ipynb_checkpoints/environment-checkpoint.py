import pickle
import numpy as np
from collections import deque
import os
import random
import cv2

def Load(file):
    f = open(file, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


class candle_class:
    def __init__(self, o,h,l,c,t):
        self.o=o
        self.h=h
        self.l=l
        self.c=c
        self.t=t

class environment:
    def __init__(self, data_dir, dlen, res_high, comm, pos_size, render =False):
        #self.data_dir = "./archive"
        self.data_dir = data_dir
        self.dlen = dlen
        self.res_high = res_high
        self.comm = comm
        self.pos_size = pos_size
        self.render=render
        if render:
            self.id = str(random.random())
            self.positions = deque(maxlen = int(dlen))
      

    def reset(self, first = False):
        self.files = [self.data_dir+"/"+x for x in os.listdir(self.data_dir) if "candle_classes" in x]
        p = random.choice(self.files)
        print("using",p)
        self.candles = Load(p)
        #print(self.candles)
        
        
        self.current_index = 0
        if first:
            self.current_index = random.randint(0,len(self.candles)-150000)
            
        self.d1_candles = deque(maxlen = self.dlen)
        self.h4_candles = deque(maxlen = self.dlen)
        self.h1_candles = deque(maxlen = self.dlen)
        self.m15_candles = deque(maxlen = self.dlen)
        self.m5_candles = deque(maxlen = self.dlen)
        
        self.position = 0
        self.entry_price = 0
        self.equity = 0
        self.current_equity = 0
        self.balance = 0
        
        
        self.last_15_open_minute = 0
        self.current_15_open_minute = 0
        
        self.last_60_open_hour = 0
        self.current_60_open_hour = 0
            
        self.get_sample_candles()
        return [self.scale_candles(self.m5_candles), self.scale_candles(self.m15_candles), self.scale_candles(self.h1_candles), self.scale_candles(self.h4_candles), self.scale_candles(self.d1_candles), self.position]
            
    def close(self):
        if self.position !=0:
            self.balance = self.equity
            self.position = 0
            self.balance -= self.pos_size * self.comm / 2
        
    def step(self, action) :
        last_equity = self.equity
        reset_entry_price = False
        if action == 1: # long
            if self.position != 1:
                self.close()
                self.position = 1
                self.balance -= self.pos_size * self.comm / 2
                reset_entry_price = True
                
        if action == 0: # short
            if self.position != -1:
                self.close()
                self.position = -1
                self.balance -= self.pos_size * self.comm / 2
                reset_entry_price = True
        if action == 2: # no position
            if self.position != 0:
                self.close()
        
        
        if self.get_sample_candles() == -1:
            print("error")
            return -1
            
        current_close = self.m15_candles[-1].c
        if reset_entry_price: self.entry_price = self.m5_candles[-1].o
        
        if self.position != 0:
            percent_change = (current_close - self.entry_price) / self.entry_price
            self.equity = self.balance + percent_change * self.pos_size * self.position
        else:
            self.equity = self.balance
        
        reward = self.equity - last_equity
        next_observation = [self.scale_candles(self.m5_candles),self.scale_candles(self.m15_candles), self.scale_candles(self.h1_candles), self.scale_candles(self.h4_candles), self.scale_candles(self.d1_candles), self.position]
            
        if self.render:
            self.plot_candles()
        return next_observation, reward, len(self.candles) == self.current_index
        
        
    def get_sample_candles(self):
        if len(self.candles) == self.current_index:
            return -1
        
        
        while True:
            # return self.dlen candles of d1, h4, h1, m15 and m5
            current_candle = self.candles[self.current_index]
            current_hour = int(current_candle.t.split(":")[0])
            open_minute = int(current_candle.t.split(":")[1])
            
            self.m5_candles.append(candle_class(current_candle.o, current_candle.h, current_candle.l, current_candle.c, str(current_hour) +":"+str(open_minute)))
            
            # m15 candles:
            self.last_15_open_minute = self.current_15_open_minute
            self.current_15_open_minute = int(open_minute/15)*15
            if self.current_15_open_minute != self.last_15_open_minute:
                self.m15_candles.append(candle_class(current_candle.o, current_candle.h, current_candle.l, current_candle.c, str(current_hour) +":"+str(open_minute)))
            else:
                if len(self.m15_candles) > 0:
                    self.m15_candles[-1].c = current_candle.c
                    self.m15_candles[-1].h = max(current_candle.h, self.m15_candles[-1].h)
                    self.m15_candles[-1].l = min(current_candle.l, self.m15_candles[-1].l)

            # h1 candles:
            self.last_60_open_hour = self.current_60_open_hour
            self.current_60_open_hour = current_hour
            if self.current_60_open_hour != self.last_60_open_hour:
                new_candle = candle_class(current_candle.o, current_candle.h, current_candle.l, current_candle.c, str(current_hour)+":00")
                self.h1_candles.append(new_candle)
            else:
                if len(self.h1_candles) > 0:
                    self.h1_candles[-1].c = current_candle.c
                    self.h1_candles[-1].h = max(current_candle.h, self.h1_candles[-1].h)
                    self.h1_candles[-1].l = min(current_candle.l, self.h1_candles[-1].l)

            # h4 candles:
            # create a new h4 candle when hour is 17, 21, 1, 5, 9, 13
            if  (current_hour == 17 or current_hour == 21 or current_hour == 1 or current_hour == 5 or current_hour == 9 or current_hour == 13) and self.current_60_open_hour != self.last_60_open_hour:
                new_candle = candle_class(current_candle.o, current_candle.h, current_candle.l, current_candle.c, str(current_hour)+":00")
                self.h4_candles.append(new_candle)
            else:
                if len(self.h4_candles) > 0:
                    self.h4_candles[-1].c = current_candle.c
                    self.h4_candles[-1].h = max(current_candle.h, self.h4_candles[-1].h)
                    self.h4_candles[-1].l = min(current_candle.l, self.h4_candles[-1].l)

            # d1 candles:
            # create a new d1 candle when hour is 17
            if  current_hour == 17 and self.current_60_open_hour != self.last_60_open_hour:
                new_candle = candle_class(current_candle.o, current_candle.h, current_candle.l, current_candle.c, str(current_hour)+":00")
                self.d1_candles.append(new_candle)
            else:
                if len(self.d1_candles) > 0:
                    self.d1_candles[-1].c = current_candle.c
                    self.d1_candles[-1].h = max(current_candle.h, self.d1_candles[-1].h)
                    self.d1_candles[-1].l = min(current_candle.l, self.d1_candles[-1].l)

            self.current_index+=1    
            if len(self.d1_candles) == self.dlen:
                break

        return self.m5_candles, self.m15_candles,  self.h1_candles, self.h4_candles, self.d1_candles
    
    
    def scale_candles(self, candles):
        def scale_p(p):
            return int((p - max_l) / hlrange * (self.res_high))
        max_h = 0
        max_l = 1000000
        for i in candles:
            if i.h > max_h:
                max_h = i.h
            if i.l < max_l:
                max_l = i.l
        hlrange = max_h - max_l
        
        
        def scale_time(t):
            hour = int(t.split(":")[0])
            minute = int(t.split(":")[1])
            total = hour * 60 + minute
            #max_t = 24*60
            #scaled = total / max_t
            scaled = total / 5
            return scaled
            
        
        
        image = []
        for i in candles:
            clm = np.zeros(shape = (self.res_high+1))
            color = 1 if i.o<i.c else -1
            high_scaled = scale_p(i.h)
            low_scaled = scale_p(i.l)
            clm[low_scaled:high_scaled] = 0.5 * color
            open_scaled = scale_p(i.o)
            close_scaled = scale_p(i.c)
            if color == 1:
                clm[open_scaled:close_scaled+1] = color
            if color == -1:
                clm[close_scaled:open_scaled+1] = color
                
            c_time = scale_time(i.t)
            clm[-1] = c_time
            image.append(clm)
        
        current_close = candles[-1].c
        scaled_close = scale_p(current_close)
        clm = np.zeros(shape = (self.res_high+1))
        clm[scaled_close] = 1
        image.append(clm)
        
        return np.array(image, dtype = "float32").T
        
        
        
   
    def plot_candles(self):
        self.positions.append(self.position)
        if len(self.positions)!=self.dlen:
            return
            
    
        def scale_p(p):
            return (p - max_l) / hlrange * h
        candles=self.m5_candles
        w = 400
        h = 300
        canvas = np.zeros((h,w,3), np.uint8) 
        l = self.dlen
        single_candle_w = w / l * 0.95
        max_h = 0
        max_l = 1000000
        for i in candles:
            if i.h > max_h:
                max_h = i.h
            if i.l < max_l:
                max_l = i.l
        hlrange = max_h - max_l

        for i in range(len(candles)):  
            
            color = (0,100,0) if self.positions[i] == 1 else (0,0,100) if self.positions[i] == -1 else (100,100,100)
            cv2.rectangle(canvas, (int(i*single_candle_w),int(scale_p(candles[i].o))), (int((i+1)*single_candle_w),int(scale_p(candles[i].c))), color, -1)
            cv2.line(canvas, (int((i+0.5)*single_candle_w),int(scale_p(candles[i].h))), (int((i+0.5)*single_candle_w),int(scale_p(candles[i].l))), color)

      
        canvas = canvas[::-1]

        cv2.imshow(self.id, canvas)
        cv2.waitKey(1)


    def __del__(self):
        if self.render:
            cv2.destroyWindow (self.id)