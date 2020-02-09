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

Training naming convention:
Date Month DD
Agent character
EnemyChar
Difficulties trained on 

E.g
January 27
Ryu
Guile
Difficulty 4, 5, and 6

Jan27-ryu-guile-456

Specifying number of episodes to train for:
- To do so the KerasRL core must be customized.
- Navigate to site-packages/rl/core.py
- On line 55, add the following parameter to the fit() signature: nb_episodes=None
- On line 124, change the while loop condition to: while self.step < nb_steps and episode < nb_episodes:
