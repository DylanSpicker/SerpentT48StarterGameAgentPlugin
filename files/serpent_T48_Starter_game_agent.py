from serpent.game_agent import GameAgent
import time
from datetime import datetime

from serpent.enums import InputControlTypes

from serpent.config import config
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from .ml_models.basic_ml_agent import BasicMLAgent

class SerpentT48_StarterGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        self.environment = self.game.environments["NUMERIC"](input_controller=self.input_controller)
        self.current_score = 0

        self.game_inputs = [{
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT"])
            }]

        self.agent = BasicMLAgent("AI2048", game_inputs=self.game_inputs)

        self.started_at = datetime.utcnow().isoformat()
        self.analytics_client.track(event_key="GAME_NAME", data={"name": "2048"})
        self.environment.new_episode(maximum_steps=5000)

    def handle_play(self, game_frame, game_frame_pipeline):
        context = self.game.api.get_context(game_frame)
        if not context:
            # Could not generate numeric board;
            # wait to render and then handle the play again
            time.sleep(0.1)
        else:
            if context['game_over']:
                context['reward'] = 0
            else:
                context['reward'] = context['current_score'] - self.current_score
                self.current_score = context['current_score']
            
            self.agent.observe(context=context)

            if not context['game_over']:
                agent_actions = self.agent.current_action
                self.environment.perform_input(agent_actions)
            else:
                self.environment.clear_input()
                self.agent.reset()
                self.environment.end_episode()
                self.input_controller.tap_key(KeyboardKey.KEY_RETURN)
                self.environment.new_episode(maximum_steps=5000, reset=self.agent.mode.name != "TRAIN")
            