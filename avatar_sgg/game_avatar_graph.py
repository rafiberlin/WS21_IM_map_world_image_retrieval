"""
    Avatar action routines
"""

from avatar_sgg.config.util import get_config
from avatar_sgg.dataset.ade20k import get_preprocessed_image_graphs_for_map_world
from avatar_sgg.game_avatar_abstract import Avatar
from avatar_sgg.image_retrieval.scene_graph_similarity_model import TextGraphVectorizer, SGEncode, get_scene_graph_encoder
import random
import os
from avatar_sgg.image_retrieval.evaluation import calculate_normalized_cosine_similarity_on_tensor
import torch


class GraphAvatar(Avatar):
    """
        The Baseline Avatar, using captioning models and BertSentences for Image Retrieval
    """

    def __init__(self, image_directory):
        config = get_config()
        if image_directory is None:
            image_directory = os.path.join(config["ade20k"]["root_dir"], "images", "training")
        config = config["game_setup"]
        self.debug = config["debug"]

        config = config["avatar"]

        self.max_number_of_interaction = config["max_number_of_interaction"]

        self._print(f"The avatar will allow only {self.max_number_of_interaction} interactions with the human player.")

        self.image_directory = image_directory
        self.map_world_preprocessed_image_graphs = get_preprocessed_image_graphs_for_map_world()
        self.current_image_graphs = None
        self.text_graph_encoder: TextGraphVectorizer = TextGraphVectorizer()
        self.img_graph_encoder: SGEncode = get_scene_graph_encoder()
        self._print(f"Avatar using Graphs for similarity.")
        self.similarity_threshold = config["similarity_threshold"]
        self.minimum_similarity_threshold = config["minimum_similarity_threshold"]
        self.aggregate_interaction = config["aggregate_interaction"]
        self._print(f"Threshold for similarity based retrieval: {self.similarity_threshold}")
        self._print(f"Aggregate Interaction: {self.aggregate_interaction}")
        self.number_of_interaction = None
        self.observation = None
        self.map_nodes = None
        self.generated_captions = None
        self.map_nodes_real_path = None
        self.vectorized_captions = None
        self.vectorized_interactions = None
        self.current_candidate_similarity = None
        self.current_candidate_ranks = None

        self.reset()

    def reset(self):
        """
        Reset important attributes for the avatar.
        :return:
        """
        self.number_of_interaction = 0
        self.observation = None
        self.map_nodes = None
        self.map_nodes_real_path = {}
        self.vectorized_captions = None
        self.vectorized_interactions = []
        self.current_candidate_similarity = 0.0
        self.current_candidate_ranks = None
        self.interactions = []
        self.room_found = False
        self.current_image_graphs = None

    def is_interaction_allowed(self):
        """
        check if the avatar is still allowed to process messages.
        :return:
        """

        if self.room_found:
            return False

        return (self.number_of_interaction < self.max_number_of_interaction)

    def get_prediction(self):
        """
        Should return the room identified by the avatar
        :return:
        """
        prediction = None

        if (self.current_candidate_ranks is not None) and (
                self.current_candidate_similarity > self.minimum_similarity_threshold):
            prediction = self.map_nodes[self.current_candidate_ranks]

        # choice = random.choice(list(self.map_nodes.items()))
        return prediction

    def __increment_number_of_interaction(self):
        self.number_of_interaction += 1

    def set_map_nodes(self, map_nodes: dict):
        """
        Only called once, when the labyrinth is initialized.
        example of entry in map_nodes:
        0: 'w/waiting_room/ADE_train_00019652.jpg'
        :param map_nodes:
        :return:
        """
        # As dictionary is sent with socket io, the int keys were converted into string.
        self.map_nodes = {int(k): map_nodes[k] for k in map_nodes.keys()}
        self.__load_image_graphs()

    def __load_image_graphs(self):

        # adding a '/' is a work around, as I preprocessed all the image graphs with the starting '/' unfortunately...
        self.current_image_graphs = {item: self.map_world_preprocessed_image_graphs['/'+item] for k, item in self.map_nodes.items()}

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
        # if "go" in message.lower():
        #     actions["move"] = self.__predict_move_action(message)
        # else:
        actions["response"] = self.__generate_response(message)

    def __set_room_found(self):

        if self.current_candidate_similarity >= self.similarity_threshold:
            self.room_found = True

    def __generate_response(self, message: str) -> str:

        message = message.lower()
        self.interactions.append(message)
        text_graphs =  self.text_graph_encoder.vectorize(self.interactions)
        if text_graphs is not None:
            self.__increment_number_of_interaction()
            for k in self.current_image_graphs.keys():
                self.current_image_graphs[k].update(text_graphs)

            self.img_graph_encoder.eval()
            test_results = []

            with torch.no_grad():
                for k, graph_dict in self.current_image_graphs.items():
                    res = self.img_graph_encoder(graph_dict)
                    test_results.append(res)
            stacked_vectors = torch.stack(test_results)


            similarity = calculate_normalized_cosine_similarity_on_tensor(stacked_vectors)
            values, ranks = torch.topk(similarity, 1, dim=0)
            values = float(values[0][0].to("cpu").numpy())
            ranks = int(ranks[0][0].to("cpu").numpy())
            if values > self.current_candidate_similarity:
                self.current_candidate_similarity = values
                self.current_candidate_ranks = ranks

            self.__set_room_found()

            found_msg = ""
            if self.room_found:
                found_msg = " I believe I found the room based on your description."
            return f"You interacted {self.number_of_interaction} times with me.{found_msg}"
        else:
            last_idx = len(self.interactions) - 1
            self.interactions.pop(last_idx)
            return "Sorry, could you try to describe the scene more precisely? (I can't infer a useful text graph.)"

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
