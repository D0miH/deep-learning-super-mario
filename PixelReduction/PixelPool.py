from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
from PIL import Image # Pillow
from math import *
import time

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

def getColor(area):
    unique, counts = np.unique(np.nonzero(area.flatten()), return_counts=True)
    #ind = np.argpartition(counts, -3)[-3:] # indices of 3 most frequent colors
    #return unique[ind].mean(), len(unique)
    return area.max()% 57, len(unique)


def pooling(array, offset_x=0, offset_y=0, size=16):
    result = np.empty([array.shape[0] // size, array.shape[1] // size], dtype=np.uint32)
    for x in range(offset_x, array.shape[0], size):
        for y in range(offset_y, array.shape[1], size):
            area = array[x:(x + size), y:(y + size)]
            if (np.count_nonzero(area) > 64): # Quarter of it beeing non-black pixel
                area_right = array[(x + size // 2):(x + size + size // 2), y:(y + size)]

                area_color, area_count = getColor(area)
                area_right_color, area_right_count = getColor(area_right)
                result[x // size, y // size] = area_color if area_count <= area_right_count else area_right_color
            else:
                result[x // size, y // size] = 0
    
    return result
        


done = True
for step in range(400):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # img = Image.fromarray(state)      # Create a PIL image
    # img.save("C:\\Users\\Daniel\\Desktop\\Text2.png")                      # View in default viewer
    if (step % 20 == 0): 
        #state = rgb_array_to_int(state)
        #state = np.array(list(int_array_to_rgb(state)), dtype=np.uint8)
        state = np.array(list(rgb_array_to_int(state)), dtype=np.uint32)
        state = state[32:] # Uppermost 32 pixels are only numbers
        state = pooling(state)
        print(state)
        time.sleep(5)

    

env.close()