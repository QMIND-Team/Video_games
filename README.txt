This is our repository for building an agent using Deep Q-Learning to play Street Fighter II

Library requirements can be installed from the requirements.txt file

To run the agent, a mode argument must be specified, either --mode train, or --mode test
TO TRAIN: python mainSF.py --mode train

To update the customized reward function, the scenario.json file in the retro library files must be changed. 
Make sure to copy the two files in the jsons folder into the "sonicthehedgehog-genisis" folder on your machine. 
These two files have the information for the custom reward function. 

Make sure you have ALL of the file in the jsons folder in the "sonicthehedgehog-genisis" folder. 
Just copy them and overwrite what you have.
