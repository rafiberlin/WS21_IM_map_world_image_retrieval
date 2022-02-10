from avatar_sgg.config.util import get_config
import collections
import pandas as pd
import string
import json
import random
import torch
import torch.utils.data as data
import os
import sng_parser

def get_ade20k_caption_annotations():
    """
    Precondition: checkout the https://github.com/clp-research/image-description-sequences under the location
    of the ade20k_dir directory
    :return: a dictionary containing the paths to the images as keys. Each image has a dictionary with a  "caption" key
    and a "category" key.
    """
    conf = get_config()["ade20k"]

    ade20k_dir = conf["root_dir"]
    ade20k_caption_dir = conf["caption_dir"]
    captions_file = os.path.join(ade20k_caption_dir, "captions.csv")
    sequences_file = os.path.join(ade20k_caption_dir, "sequences.csv")
    captions_df = pd.read_csv(captions_file, sep="\t", header=0)
    sequences_df = pd.read_csv(sequences_file, sep="\t", header=0)
    sequences_df["d1"] = sequences_df["d1"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d2"] = sequences_df["d2"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d3"] = sequences_df["d3"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d4"] = sequences_df["d4"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d5"] = sequences_df["d5"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["merged_sequences"] = sequences_df[["d1", "d2", "d3", "d4", "d5"]].agg(lambda x: ''.join(x.values),
                                                                                        axis=1).T
    sequences_fram = sequences_df[["image_id", "image_path", "image_cat", "merged_sequences"]]
    captions_df = pd.merge(captions_df, sequences_fram, how='inner', left_on=['image_id'], right_on=['image_id'])
    captions_df["image_path"] = captions_df["image_path"].map(
        lambda a: os.path.join("file://", ade20k_dir, "images", a))
    captions_df.drop(["Unnamed: 0"], axis=1)

    captions_list = [{"image_id": row["image_id"], "id": row["caption_id"], "caption": row["caption"],
                      "image_path": row["image_path"], "image_cat": row["image_cat"],
                      "merged_sequences": row["merged_sequences"]} for i, row in captions_df.iterrows()]
    # { id: list(captions_df[captions_df["image_id"] == id ]["caption"]) for id in ids  }

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(dict)
    for val in captions_list:
        caption = val['caption']
        category = val['image_cat']
        image_path = val["image_path"]
        merged_sequences = val["merged_sequences"]
        image_path_to_caption[image_path]["category"] = category
        image_path_to_caption[image_path]["merged_sequences"] = merged_sequences
        if "caption" not in image_path_to_caption[image_path].keys():
            image_path_to_caption[image_path]["caption"] = [caption]
        else:
            image_path_to_caption[image_path]["caption"].append(caption)

    return image_path_to_caption


def get_ade20k_split(test_proportion: int = 15, test_size: int = 10):
    """
    Returns train, dev and test split.
    Dev has only one image.
    TODO: probably better to use cross validation for the splits
    :param test_proportion:
    :return:
    """
    assert test_proportion > 0 and test_proportion < 100
    captions = get_ade20k_caption_annotations()
    # Make the split consistent
    random.seed(1)
    keys = list(captions.keys())
    random.shuffle(keys)
    start_idx = test_size
    dev = {k: captions[k] for k in keys[:test_size]}
    size = len(keys[start_idx:])
    test_idx = int(test_proportion * size / 100)
    test = {k: captions[k] for k in keys[start_idx:test_idx]}
    train = {k: captions[k] for k in keys[test_idx:]}
    return train, dev, test


def get_categories(split):
    cat = {}
    one_key = list(split.keys())[0]
    if "category" in split[one_key].keys():
        cat = {i: split[k]["category"] for i, k in enumerate(split)}
    return cat

def group_entry_per_category(category):
    category_to_entry_lookup = collections.defaultdict(list)
    for k, v in category.items():
        category_to_entry_lookup[v].append(k)




class SGEncodingADE20KMapWorldInstances(data.Dataset):
    """ SGEncoding dataset """

    def __init__(self):
        super(SGEncodingADE20KMapWorldInstances, self).__init__()

        conf = get_config()["scene_graph"]
        cap_graph_file = conf["capgraphs_file"]
        vg_dict_file = conf["visual_genome_dict_file"]
        ade20k_map_world_preprocessed_img_graph_file  = conf["ade20k_map_world_preprocessed_img_graph"]

        cap_graph = json.load(open(cap_graph_file))
        vg_dict = json.load(open(vg_dict_file))

        self.img_sg = json.load(open(ade20k_map_world_preprocessed_img_graph_file))
        self.key_list = list(self.img_sg.keys())

        # generate union predicate vocabulary
        self.sgg_rel_vocab = list(set(cap_graph['idx_to_meta_predicate'].values()))
        self.txt_rel_vocab = list(set(cap_graph['cap_predicate'].keys()))

        # generate union object vocabulary
        self.sgg_obj_vocab = list(set(vg_dict['idx_to_label'].values()))
        self.txt_obj_vocab = list(set(cap_graph['cap_category'].keys()))

        # vocabulary length
        self.num_sgg_rel = len(self.sgg_rel_vocab)
        self.num_txt_rel = len(self.txt_rel_vocab)
        self.num_sgg_obj = len(self.sgg_obj_vocab)
        self.num_txt_obj = len(self.txt_obj_vocab)


    def get(self, key):
        return self.img_sg[key]


    def _to_tensor(self, inp_dict):
        return {'entities': torch.LongTensor(inp_dict['entities']),
                'relations': torch.LongTensor(inp_dict['relations'])}

    def _generate_tensor_by_idx(self, idx):
        img = self._to_tensor(self.img_sg[self.key_list[idx]]['img'])
        img_graph = torch.FloatTensor(self.img_sg[self.key_list[idx]]['image_graph'])
        img['graph'] = img_graph

        # txt = self._to_tensor(self.img_txt_sg[self.key_list[idx]]['txt'])
        # txt_graph = torch.FloatTensor(self.img_txt_sg[self.key_list[idx]]['text_graph'])
        # txt['graph'] = txt_graph
        return img

    def __getitem__(self, item):
        fg_img = self._generate_tensor_by_idx(item)

        return fg_img

    def __len__(self):
        return len(self.key_list)

    def generate_text_graph(self, captions):
        raw_graphs = None

        if type(captions) is list:
            raw_graphs = [sng_parser.parse(cap) for cap in captions]
        elif type(captions) is str:
            raw_graphs = [sng_parser.parse(captions)]
        else:
            assert raw_graphs is not None

        cleaned_graphs = []
        for i, g in enumerate(raw_graphs):
            entities = g["entities"]
            relations = g["relations"]
            filtered_entities = [e["lemma_head"] if e["lemma_head"] in self.txt_obj_vocab else 'none' for e in
                                 entities]
            filtered_relations = [[r["subject"], r["object"], r["lemma_relation"]] for r in relations if
                                  r["lemma_relation"] in self.txt_rel_vocab]

            extracted_graph = {'entities': filtered_entities, 'relations': filtered_relations}
            cleaned_graphs.append(extracted_graph)


        return cleaned_graphs



if __name__ == "__main__":
    print("Start")
    train, dev, test = get_ade20k_split()
    # print(f"Train Split: {len(train)}")
    # print(f"Dev Split: {len(dev)}")
    # print(f"Test Split: {len(test)}")

    map_world_dataset = SGEncodingADE20KMapWorldInstances()
    pass


    print("Done")
