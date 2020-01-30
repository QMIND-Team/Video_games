from rl.core import Processor
from preprocessing import preprocess

class CNNProcessor(Processor):

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        return preprocess(observation)

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        return action

    def process_state_batch(self, batch):
        return batch

    @property
    def metrics(self):
        return []

    @property
    def metrics_names(self):
        return []