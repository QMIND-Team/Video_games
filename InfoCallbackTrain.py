import tensorflow as tf
from trainingMetrics import save_wins


class InfoCallbackTrain(tf.keras.callbacks.Callback):
    def __init__(self, state):
        self.player_win = False
        self.state = state

    def on_step_end(self, step, logs=None):

        if (logs['info'])['matches_won'] == 8:
            self.player_win = True

    def on_episode_end(self, episode, logs=None):
        
        print("Train Episode {} Win: {}".format(episode + 1, self.player_win))

        if self.player_win == False:
            save_wins(False, "train", self.state)
        elif self.player_win == True:
            save_wins(True, "train", self.state)
            self.player_win = False
