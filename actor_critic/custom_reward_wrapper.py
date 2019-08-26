from math import inf

import gym


class RewardWrapper(gym.Wrapper):
    """
    Takes the mario gym environment and applies a custom reward function.
    """

    def __init__(self, env):
        self.env = env.unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.env.reward_range = (-inf, inf)

        # start value of mario is 40
        self.last_x_pos = 40
        self.max_x_pos = 40

        # start time is 400
        self.last_time_value = 400

        self.last_score = 0

        self.last_status = "small"

    def position_reward(self, x_pos):
        """
        Rewards mario for going right and punishes him for going left.
        :return:    The reward value
        """
        reward = x_pos - self.last_x_pos

        if x_pos < self.max_x_pos:
            return 0

        self.max_x_pos = x_pos

        # reduce negative reward and clip the reward to max 5
        if reward < 0:
            reward = 0
        elif reward > 1:
            reward = 1

        self.last_x_pos = x_pos

        return reward

    def time_penalty(self, time):
        """
        Punishes mario for doing nothing.
        :param time: The currently remaining time
        :return:    The negative reward value
        """
        reward = time - self.last_time_value

        self.last_time_value = time

        return reward

    def death_penalty(self):
        """
        Punishes mario for dying or letting the time run out.
        :return: The negative reward
        """
        if self.env._is_dying or self.env._is_dead:
            return -25

        return 0

    def score_reward(self, score):
        """
        Rewards mario for increasing the ingame score.
        :param score:
        :return: The reward
        """
        score_delta = score - self.last_score

        self.last_score = score

        return score_delta / 5

    def status_reward(self, status):
        """
        Rewards mario for collecting a mushroom and getting tall or collecting a fire flower.
        Mario gets punished for loosing the fire flower or getting small again.
        :param status: The current status ("small", "tall" or "fireball")
        :return: The reward
        """
        if status == self.last_status or (
                (self.last_status == "tall" or self.last_status == "fireball") and self.env._is_dying):
            # if the status is the same or mario died because of no time don't punish him
            return 0

        if self.last_status == "small" and (status == "tall" or status == "fireball"):
            # give mario a reward for collecting a mushroom or a flower
            self.last_status = status
            return 25
        elif (self.last_status == "tall" or self.last_status == "fireball") and status == "small":
            # punish mario for going small
            self.last_status = status
            return -10

    def step(self, action):
        next_state, _, done, info = self.env.step(action)

        position_reward = self.position_reward(info["x_pos"])
        score_reward = self.score_reward(info["score"])
        status_reward = self.status_reward(info["status"])
        #time_penalty = self.time_penalty(info["time"])
        death_penalty = self.death_penalty()
        reward = position_reward + score_reward + status_reward + death_penalty

        return next_state, reward, done, info

    def reset(self, **kwargs):
        # reset the position variables
        self.last_x_pos = 40
        self.max_x_pos = 40

        # reset the time penalty variables
        self.last_time_value = 400

        # reset the score
        self.last_score = 0

        # reset the status
        self.last_status == "small"

        return self.env.reset()
