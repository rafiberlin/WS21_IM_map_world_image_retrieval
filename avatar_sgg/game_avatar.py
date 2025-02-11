"""
    Avatar action routines
"""

from avatar_sgg.config.util import get_config
import random


DIRECTION_TO_WORD = {
    "n": "north",
    "e": "east",
    "w": "west",
    "s": "south"
}


def direction_to_word(direction: str):
    if direction in DIRECTION_TO_WORD:
        return DIRECTION_TO_WORD[direction]
    return direction


def directions_to_sent(directions: str):
    if not directions:
        return "nowhere"
    n = len(directions)
    if n == 1:
        return direction_to_word(directions[0])
    words = [direction_to_word(d) for d in directions]
    return ", ".join(words[:-1]) + " or " + words[-1]


class Avatar(object):
    """
        The avatar_sgg methods to be implemented
    """

    def step(self, observation: dict) -> dict:
        """
        The current observation for the avatar_sgg.

        For new player messages only the 'message' will be given.
        For new situations the 'image' and 'directions' will be given.

        The agent should return a dict with "move" or "response" keys.
        The response will be sent to the player.
        The move command will be executed by the game master.
        Possible move commands are: {"n": "north", "e": "east", "w": "west", "s": "south"}

        :param observation: {"image": str, "directions": [str], "message": str }
        :return: a dict with "move" and/or "response" keys; the dict could also be empty to do nothing
        """
        raise NotImplementedError("step")

    def is_interaction_allowed(self):
        """
        Depends on the number of interactions allowed per game
        :return:
        """
        raise NotImplementedError("is_interaction_allowed")

    def get_prediction(self):
        """
        Should return the room identified by the avatar
        :return:
        """
        raise NotImplementedError("get_prediction")

class SimpleAvatar(Avatar):
    """
        The simple avatar_sgg is only repeating the observations.
    """

    def __init__(self, image_directory):

        config = get_config()["game_setup"]

        self.debug = config["debug"]
        config = config["avatar"]

        self.max_number_of_interaction = config["max_number_of_interaction"]

        if self.debug:
            print(f"The avatar will allow only {self.max_number_of_interaction} interactions with the human player.")

        self.number_of_interaction = 0

        self.image_directory = image_directory
        self.observation = None
        self.map_nodes = None

    def is_interaction_allowed(self):
        """
        check if the avatar is still allowed to process messages.
        :return:
        """
        return self.number_of_interaction < self.max_number_of_interaction


    def get_prediction(self):
        """
        Should return the room identified by the avatar
        :return:
        """
        #TODO Replace it by the results of the models in use.
        pass
        choice =  random.choice(list(self.map_nodes.items()))
        return choice

    def __increment_number_of_interaction(self):
        self.number_of_interaction += 1

    def set_map_nodes(self, map_nodes: dict):
        """
        Only called once, when the labyrinth is initialized.
        :param map_nodes:
        :return:
        """
        self.map_nodes = map_nodes

    def step(self, observation: dict) -> dict:
        if self.debug:
            print(observation)
        actions = dict()
        if observation["image"]:
            self.__update_observation(observation)
        if observation["message"]:
            self.__update_actions(actions, observation["message"])
        return actions

    def __update_observation(self, observation: dict):
        self.observation = observation

    def __update_actions(self, actions, message):
        if "go" in message.lower():
            actions["move"] = self.__predict_move_action(message)
        else:
            actions["response"] = self.__generate_response(message)

    def __generate_response(self, message: str) -> str:
        self.__increment_number_of_interaction()
        message = message.lower()

        if message.startswith("what"):
            if self.observation:
                return "I see " + self.observation["image"]
            else:
                return "I dont know"

        if message.startswith("where"):
            if self.observation:
                return "I can go " + directions_to_sent(self.observation["directions"])
            else:
                return "I dont know"

        if message.endswith("?"):
            if self.observation:
                return "It has maybe something to do with " + self.observation["image"]
            else:
                return "I dont know"

        return f"You interacted {self.number_of_interaction} times with me."

    def __predict_move_action(self, message: str) -> str:
        if "north" in message:
            return "n"
        if "east" in message:
            return "e"
        if "west" in message:
            return "w"
        if "south" in message:
            return "s"
        return "nowhere"
