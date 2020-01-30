This is our repository for building an agent using Deep Q-Learning to play Street Fighter II

Library requirements can be installed from the requirements.txt file

To run the agent, a mode argument must be specified, either --mode train, or --mode test

To customize the reward function, the scenario.json file in the retro library files must be changed.
The default value gives a reward based on the player's score.
To reward based on the difference in player's health, paste the following in the reward>variables object in the
scenario file located at: site-packages/retro/data/stable/stable/StreetFighterIISpecialChampionEdition-Genesis/scenario.json

    "health": {
        "penalty": 1.0
    },
    "enemy_health": {
        "penalty": -1.0
    }
    
Note that in the infocallbacktrain and infocallback test files the value that measures a win will be 2 for Ted and 8 for everyone else. If this value is not adjusted wins will be measured incorrectly. 

To properly enable testing, paste the following snippet into: Lib\site-packages\rl\policy.py

class BoltzmannQPolicyTest(Policy):
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1, clip=(-500., 500.)):
        super(BoltzmannQPolicyTest, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicyTest, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config
