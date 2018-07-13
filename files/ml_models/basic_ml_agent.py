from serpent.machine_learning.reinforcement_learning.agent import Agent

from serpent.game_frame import GameFrame
from serpent.game_frame_buffer import GameFrameBuffer
from serpent.input_controller import KeyboardEvent, MouseEvent
from serpent.enums import InputControlTypes

from serpent.utilities import SerpentError

from collections import deque

import os
import io
import enum
import random
import json
import pickle

import numpy as np
import h5py

import skimage.io
import skimage.util

try:
    from .keras_dqn_model import KerasDQN
except ImportError:
    raise SerpentError("Setup has not been been performed for the ML module. Please run 'serpent setup ml'")


class BasicAgentModes(enum.Enum):
    OBSERVE = 0
    TRAIN = 1
    EVALUATE = 2

        
class BasicMLAgent(Agent):

    def __init__(
        self,
        name,
        game_inputs=None,
        callbacks = None,
        observe_steps=5000
    ):
        super().__init__(name, game_inputs=game_inputs, callbacks=callbacks)
        
        # Basic Model Parameters
        self.gamma = 0.99                              # Discount Rate
        self.epsilon = 0.95                             # Exploration Parameter for Greedy Epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.001

        self.mini_batches = 100
        self.mini_batch_size = 128
        self.observe_steps = observe_steps              # How long to run observations for
        self.observe_steps_remaining = observe_steps

        self.memory = deque(maxlen=35000)   # Keep an Instance of the {State, Action, Reward, Next_State, Terminal} pairs in Memory
        self.mode = BasicAgentModes.OBSERVE # Set the mode to the 'observe' on initialization
        self.observe_mode = "RANDOM"        # Set the observe mode to 'random' by default; can switch to 'model'
        self.dqn = KerasDQN(input_size=(4,4,1), output_size=len(game_inputs[0]["inputs"]))
        self.current_score = 0
        self.final_scores = []

        # Uncomment the following lines if you wish to load the memory/weights from a previous training session
        # self.memory = pickle.load( open( "starter_sqn_memory.p", "rb" ) )
        # self.dqn.model.load_weights("starter_dqn_weights.h5")

        # Default Values
        self.current_state = np.zeros(shape=(4,4,1), dtype=np.float32)
        self.current_action_pred = 0
        
    def observe(self, context, **kwargs):
        if self.observe_steps_remaining % 2500 == 0 and self.final_scores != []:
            print("\n")
            print("Steps Remaining in Episode:", self.observe_steps_remaining)
            print("\t Current Epsilon: ", str(self.epsilon))
            print("\t Max Game Score: ", str(max(self.final_scores)))
            print("\t Current Game Score: ", str(self.current_score))
            print("\t Average Game Score: ", str(sum(self.final_scores)/len(self.final_scores)))

        if context['game_over']:
            next_state = None
            if self.current_score > 0:
                self.final_scores += [self.current_score]
                self.current_score = 0
        else:
            next_state = context['numeric_matrix'].reshape((4,4,1))
            self.current_score = context['current_score']

        self._remember(self.current_state, self.current_action_pred, context['reward'], next_state, context['game_over'])
        
        if next_state is None:
            self.current_state = np.zeros(shape=(4,4,1), dtype=np.float32)
            self.current_action = 0
        else:
            self.current_state = (next_state.astype(np.float32)/4096).reshape((4,4,1))
            self.current_action = self._generate_actions()
            self.observe_steps_remaining -= 1

            if self.observe_steps_remaining == 0:
                for _ in range(self.mini_batches):
                    self._replay(self.mini_batch_size)


                if self.mode == BasicAgentModes.OBSERVE and self.observe_mode == "MODEL":
                    self.epsilon *= self.epsilon_decay
                    self.epsilon = max(self.epsilon, self.min_epsilon)

                self.observe_steps_remaining = self.observe_steps
                self.observe_mode = "MODEL"

                # Save Current Training
                self.dqn.model.save_weights("starter_dqn_weights.h5")
                pickle.dump( self.memory, open( "starter_sqn_memory.p", "wb" ) )

    def _generate_actions(self):
        if self.mode == BasicAgentModes.OBSERVE and self.observe_mode == "RANDOM":
            # Generate Random Action
            self.current_action_pred = random.randint(0, len(self.game_inputs[0]["inputs"])-1)
        elif self.mode == BasicAgentModes.OBSERVE and self.observe_mode == "MODEL":
            if np.random.rand() <= self.epsilon:
                self.current_action_pred = random.randint(0, len(self.game_inputs[0]["inputs"])-1)
            else:
                preds_ = self.dqn.model.predict(np.array([self.current_state]))
                self.current_action_pred = np.argmax(preds_[0])

        actions = list()

        label = self.game_inputs_mappings[0][self.current_action_pred]
        action = self.game_inputs[0]["inputs"][label]

        actions.append((label, action, None))

        for action in actions:
            self.analytics_client.track(
                event_key="AGENT_ACTION",
                data={
                    "label": action[0],
                    "action": [str(a) for a in action[1]],
                    "input_value": action[2]
                }
            )

        return actions

    def _replay(self, batch_size=32):
        batch = random.sample(self.memory, batch_size)
        
        for current_state, action, reward, next_state, terminal in batch:
            pred_target = reward
            next_state = np.array([next_state])
            current_state = np.array([current_state])
            
            if not terminal:
                pred_target += self.gamma*np.argmax(self.dqn.model.predict(next_state)[0])
            
            target = self.dqn.model.predict(current_state)
            target[0][action] = pred_target
            self.dqn.model.fit(current_state, target, epochs=1, verbose=0)

    def _remember(self, state, action, reward, next_state, terminal):
        if state is not None:
            if next_state is None:
                terminal = True
            self.memory.append((state, action, reward, next_state, terminal))
