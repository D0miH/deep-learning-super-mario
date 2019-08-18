from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
from PIL import Image # Pillow
from math import *

class PixelConverter:
    def __init__(self):
        self.dict = {}
        self.dict[self._getKey(252, 252, 252)] = 0
        self.dict[self._getKey(104, 136, 252)] = 0
        self.dict[self._getKey(0, 168, 0)] = 0
        self.dict[self._getKey(184, 248, 24)] = 0

    def _getKey(self, r, g, b):
        return (r * 255 + g) * 255 + b

    def _getMax(self):
        return max(self.dict.values())
    
    def convert(self, r, g, b):
        key = self._getKey(r, g, b)
        if key in self.dict:
            return self.dict[key]
        
        value = self._getMax() + 1
        self.dict[key] = value
        return value

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, RIGHT_ONLY)

converter = PixelConverter()

def rgb_array_to_int(array):
    for x in array:
        yield [ y[2] + (y[1] + y[0] * 255) * 255 for y in x ]

#def rgb_array_to_int(array):
#    for x in array:
#        yield [ converter.convert(*y) for y in x ]

def int_array_to_rgb(array):
    colors = [
        [0, 0, 0],
        [183, 28, 28],
        [183, 157, 28],
        [79, 183, 28],
        [28, 183, 105],
        [28, 131, 183],
        [53, 28, 183]
    ]

    for x in array:
        yield [ colors[y % len(colors)] for y in x]

def pooling(array):
    result = np.empty([array.shape[0] // 16, array.shape[1] // 16], dtype=np.uint32)
    for x in range(0, array.shape[0], 16):
        for y in range(0, array.shape[1], 16):
            area = array[x:(x + 16), y:(y + 16)]
            if (np.count_nonzero(area) > 64): # Quarter of it beeing non-black pixel
                result[x // 16, y // 16] = np.max(area)
            else:
                result[x // 16, y // 16] = 0
    
    return result
        


done = True
for step in range(1000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if (step % 20 == 0): 
        #state = rgb_array_to_int(state)
        #state = np.array(list(int_array_to_rgb(state)), dtype=np.uint8)
        state = np.array(list(rgb_array_to_int(state)), dtype=np.uint8)
        state = state[32:] # Uppermost 32 pixels are only numbers
        state = pooling(state)
        print(state)
        #img = Image.fromarray(state)      # Create a PIL image
        #img.save("C:\\Users\\Daniel\\Desktop\\Text2.png")                      # View in default viewer
    

env.close()