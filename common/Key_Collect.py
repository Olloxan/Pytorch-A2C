import numpy as np
import random
import gym
from gym import spaces



class Key_Collect():

    def __init__(self,dim_room=(10,10), max_steps=120, num_keys=1):        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, *dim_room), dtype=np.float32)        
        self.action_space = spaces.Discrete(5)

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.reward_get_key = 1
        self.reward_finished = 10
        self.reward_last = 0
         
        # tile Values
        self.empty_tile = 1
        self.player = 4
        self.wall = 0
        self.key = 2
        self.goal = 3
        self.player_in_goal = 7
        self.player_on_key = 6

        # game init              
        self.num_keys = num_keys
        self.max_steps = max_steps

        # game state
        self.frame = 0   
        self.current_num_keys = num_keys
        self.last_num_keys = num_keys
        self.dim_room = dim_room
                        
        # status flags
        self.player_not_all_keys = 0
        self.player_has_key = 1
        
        # player state
        self.player_state = 0 # 0, if player was not at Key_position yet, 1, else

        self.env = self.reset()
     
    def step(self, action):
        self.frame += 1
        self.move_player(action)
        reward = self.calculate_reward()
        self.update_player_key_state()        
        done = self.check_if_done()        
        return self.render(), reward, done, {}    

    def calculate_reward(self):
        reward = self.penalty_for_step        
        if self.last_num_keys - self.current_num_keys != 0:
            reward += 1
            self.last_num_keys = self.current_num_keys
        player_pos = self.convert_pos(np.where(self.env > 3))      
        if self.is_player_in_goal(player_pos) and not self.has_player_no_key():
            reward += 10   
        return reward
    
    def reset(self):        
        # generate empty field
        room = np.ones(self.dim_room)       
        self.set_obstacle(room)     # 0        
        self.set_goal(room)         # 3        
        self.set_keys(room)          # 2        
        self.set_player(room)       # 4

        self.env = room
        self.frame = 0
        self.player_state = 0
        self.current_num_keys = self.num_keys
        self.last_num_keys = self.num_keys
        return self.render()    

    def set_obstacle(self, room):      
        horizontal_obstacle = random.randint(0,1)
        num_passages = 3
        if horizontal_obstacle:          
            passages = random.sample(range(room.shape[1]), num_passages)
            lower_boundery = 2
            upper_boundery = room.shape[0] - 3           
            obstacle_row = random.randint(lower_boundery, upper_boundery)
            room[obstacle_row,:] = self.wall
            room[obstacle_row, passages] = 1
        else:           
            passages = random.sample(range(room.shape[0]), num_passages)
            lower_boundery = 2
            upper_boundery = room.shape[1] - 3           
            obstacle_col = random.randint(lower_boundery, upper_boundery)
            room[:,obstacle_col] = self.wall
            room[passages, obstacle_col] = 1       
        return room
           
    def set_goal(self, room):        
        pos = self.get_random_room_position(room)        
        room[pos[0],pos[1]] = self.goal                
        return room

    def set_keys(self, room):
        if self.num_keys > 5:
            raise ValueError('not more than 5 keys')
        for i in range(self.num_keys):
            pos = self.get_random_room_position(room)        
            room[pos[0],pos[1]] = self.key           
        return room

    def set_player(self, room):
        pos = self.get_random_room_position(room)        
        room[pos[0],pos[1]] = self.player          
        return room

    def get_random_room_position(self, room):
        room_dims = 2
        pos = np.zeros(room_dims, dtype=np.int32)
        result = 0
        while result != 1:            
            for i in range(room_dims):               
                pos[i] = random.randint(0, room.shape[i]-1)             
            result = room[pos[0], pos[1]]
        return pos        

    def move_player(self, action):
        player_pos = self.convert_pos(np.where(self.env > 3)) # pos[0] = y, pos[1] = x                        
        action_definition = [[0, 0], # nix
                             [0, 1], # rechts
                             [1, 0], # unten 
                             [0, -1],# links 
                             [-1, 0]]# oben
        
        if self.is_player_in_goal(player_pos) and self.has_player_no_key():
            return        
        
        player_pos_x = player_pos[0] + action_definition[action][0]
        player_pos_y = player_pos[1] + action_definition[action][1]
        if self.out_of_room_boundaries(player_pos_y, player_pos_x):
            return
            
        if self.env[player_pos_x][player_pos_y] == self.wall:
            return

        self.env[player_pos[0]][player_pos[1]] = self.empty_tile    # tile where player moved from
        if self.env[player_pos_x][player_pos_y] == self.key:           
           self.current_num_keys -= 1
           if self.is_last_key():
                self.env[player_pos_x][player_pos_y] = self.player_on_key
           else:
               self.env[player_pos_x][player_pos_y] = self.player        
        elif self.env[player_pos_x][player_pos_y] == self.goal:
            self.env[player_pos_x][player_pos_y] = self.player_in_goal
        else:
            self.env[player_pos_x][player_pos_y] = self.player        
    
    def is_last_key(self):                
        return len(np.where(self.env == self.key)[0]) == 1
    
    def has_player_no_key(self):
        return self.player_state == self.player_not_all_keys

    def is_player_in_goal(self, player_pos):
        return self.env[player_pos[0]][player_pos[1]] == self.player_in_goal
    
    def update_player_key_state(self):
        player_on_key_pos = np.where(self.env == self.player_on_key)        
        if player_on_key_pos[0] != {}:
            player_on_key_pos = self.convert_pos(player_on_key_pos)
            self.player_state = self.player_has_key
            self.env[player_on_key_pos[0]][player_on_key_pos[1]] = self.player


    def out_of_room_boundaries(self, col_index, row_index):
        return row_index < 0 or row_index > self.dim_room[0] -1 or col_index < 0 or col_index > self.dim_room[1] -1                 

    def convert_pos(self, player_pos):
        return [player_pos[0][0], player_pos[1][0]]        
    
    def check_if_done(self):        
        max_steps_reached = self.frame > self.max_steps
        player_pos = self.convert_pos(np.where(self.env > 3))        
        game_won = self.is_player_in_goal(player_pos) and self.player_state == 1
        return max_steps_reached or game_won

    def render(self):
        img = self.room_to_rgb()
        return img

    def room_to_rgb(self):
        height = self.dim_room[0]
        width = self.dim_room[1]
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        for x in range(height):
            for y in range(width):
                if self.env[x, y] == self.wall:
                    rgb[x, y] = [0, 0, 0]
                elif self.env[x, y] == self.empty_tile:
                    rgb[x, y] = [0, 0, 1]
            
                elif self.env[x, y] == self.key:
                    rgb[x, y] = [0, 1, 1]   
                elif self.env[x, y] == self.goal:
                    rgb[x, y] = [1, 0, 0]
                elif self.env[x, y] == self.player and self.player_state == self.player_not_all_keys:
                    rgb[x, y] = [0, 1, 0]
                elif self.env[x, y] == self.player and self.player_state == self.player_has_key:
                    rgb[x, y] = [1, 1, 0]
                elif self.env[x,y] == self.player_in_goal:
                    rgb[x,y] = [1, 0, 1]
        rgb = rgb.transpose(2,0,1)
        return rgb

# creates a KeyCollect environment of size (3 x width x height) with (8x8) tiles instead of (1x1) tiles     
class Key_Collect_upscaled(Key_Collect):
    def __init__(self, dim_room = (10, 10), max_steps = 120, num_keys = 1):
        super(Key_Collect_upscaled, self).__init__(dim_room, max_steps, num_keys)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, dim_room[0]*8, dim_room[1]*8), dtype=np.float32)

    def reset(self):
        image = super(Key_Collect_upscaled, self).reset()
        return self.upscale(image)

    def step(self, action):
        image, reward, done, _ = super(Key_Collect_upscaled, self).step(action)
        return self.upscale(image), reward, done, {}
    
    def upscale(self,input):
        upscaled = np.zeros((3,*dim_room))
        for i in range(upscaled.shape[0]):
            for j in range(upscaled.shape[1]):
                for k in range(upscaled.shape[2]):            
                    upscaled[i][j][k]=input[i][j//8][k//8]
        return upscaled

# debug configuration of environment KeyCollect, creates always the same environment
class Key_Collect_debug(Key_Collect):
    def __init__(self, dim_room = (10, 10), max_steps = 120, num_keys = 1, rnd_seed=1001):
        super(Key_Collect_debug, self).__init__(dim_room, max_steps, num_keys)
        random.seed(rnd_seed)
        self.random_state = random.getstate()        

    def reset_debug(self):         
        random.setstate(self.random_state)
        image = super(Key_Collect_debug, self).reset()
        return image
   
