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
import numpy as np


def get_ade20k_caption_annotations(path_prefix=None):
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
    if path_prefix is None:
        print("Using Real Image Paths as Key.")
        captions_df["image_path"] = captions_df["image_path"].map(
            lambda a: os.path.join("file://", ade20k_dir, "images", a))
    else:
        captions_df["image_path"] = captions_df["image_path"].map(
            lambda a: os.path.join(path_prefix, a))
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


def get_ade20k_split(test_proportion: int = 15, test_size: int = 10, path_prefix=None):
    """
    Returns train, dev and test split.
    Dev has only one image.
    TODO: probably better to use cross validation for the splits
    :param test_proportion:
    :return:
    """
    assert test_proportion > 0 and test_proportion < 100
    captions = get_ade20k_caption_annotations(path_prefix)
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

    def generate_text_graph(self, captions):
        raw_graphs = None

        if type(captions) is list:
            raw_graphs = [sng_parser.parse(cap) for cap in captions]
        elif type(captions) is str:
            raw_graphs = [sng_parser.parse(captions)]
        else:
            assert raw_graphs is not None


def output_split_list_with_new_prefix(split, old, new, file_path):
    """

    :param split:
    :param old: old prefix
    :param new: new prefix
    :param file_path: where to write the file
    :return:
    """
    prefix_index_end = len(old)

    new_paths = []
    for k in split.keys():
        idx_start = k.find(old)
        new_paths.append(new + k[idx_start + prefix_index_end:])

    with open(file_path, 'w') as outfile:
        json.dump(new_paths, outfile)

    print("Saved", file_path)


def generate_text_graph(split, output_path, caption_number=None):
    if not os.path.isfile(output_path):
        text_graphs = {}
        conf = get_config()["scene_graph"]
        cap_graph_file = conf["capgraphs_file"]
        cap_graph = json.load(open(cap_graph_file))
        txt_rel_vocab = list(set(cap_graph['cap_predicate'].keys()))
        txt_rel2id = {key: i + 1 for i, key in enumerate(txt_rel_vocab)}
        txt_obj_vocab = list(set(cap_graph['cap_category'].keys()))
        txt_obj2id = {key: i + 1 for i, key in enumerate(txt_obj_vocab)}

        # generate union object vocabulary
        txt_obj_vocab = list(set(cap_graph['cap_category'].keys()))

        for k in split.keys():

            if caption_number is not None:
                captions = split[k]["caption"][caption_number]
            else:
                captions = split[k]["caption"]

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
                filtered_entities = [e["lemma_head"] if e["lemma_head"] in txt_obj_vocab else 'none' for e in
                                     entities]
                filtered_relations = [[r["subject"], r["object"], r["lemma_relation"]] for r in relations if
                                      r["lemma_relation"] in txt_rel_vocab]

                extracted_graph = {'entities': filtered_entities, 'relations': filtered_relations}
                cleaned_graphs.append(extracted_graph)

            encode_txt = {'entities': [], 'relations': []}

            for item in cleaned_graphs:
                entities = [txt_obj2id[e] for e in item['entities']]
                relations = [[entities[r[0]], entities[r[1]], txt_rel2id[r[2]]] for r in item['relations']]
                encode_txt['entities'] = encode_txt['entities'] + entities
                encode_txt['relations'] = encode_txt['relations'] + relations

            # === for text_graph =============================================here
            entities = encode_txt['entities']
            relations = encode_txt['relations']
            if len(relations) == 0:
                txt_graph = np.zeros((len(entities), 1))
            else:
                txt_graph = np.zeros((len(entities), len(relations)))

            text_graph = []
            for i, es in enumerate(entities):
                for j, rs in enumerate(relations):
                    if es in rs:
                        txt_graph[i, j] = 1
                    else:
                        txt_graph[i, j] = 0

            text_graph.append(txt_graph.tolist())

            text_graphs[k] = {
                'txt': encode_txt,
                'text_graph': text_graph,
                'category': split[k]["category"]}#needed later to perform the category based recall

        with open(output_path, 'w') as outfile:
            print("Saving Text Graphs under:", output_path)
            json.dump(text_graphs, outfile)
    else:
        print("Loading:", output_path)
        text_graphs = json.load(open(output_path))

    return text_graphs

def get_preprocessed_text_text_graphs_for_test():
    """
    This function returns the captions of the ADE20K test sets, as graph. They are not merged
    and are available as tuple in the "entry" key.
    :return:
    """

    conf = get_config()
    _, _, test = get_ade20k_split(path_prefix="images")
    txt_graphs_1 = generate_text_graph(test, conf["scene_graph"]["ade20k_text_graph_1"], 0)
    txt_graphs_2 = generate_text_graph(test, conf["scene_graph"]["ade20k_text_graph_2"], 1)

    txt_keys = list(txt_graphs_1.keys())

    txt_graphs = {}
    for k in txt_keys:
        item = txt_graphs_1[k]
        item2 =  txt_graphs_2[k]
        if len(item["txt"]['entities']) < 2 \
                or len(item2["txt"]["entities"]) < 2 \
                or len(item["txt"]['relations']) < 1 \
                or len(item2["txt"]['relations']) < 1:
            print("no relationship detected, skipping:", k)
            continue
        else:
            txt_graphs[k] = {"entry": (item, item2), "category": item["category"]}

    return txt_graphs


def get_preprocessed_image_text_graphs_for_test():
    """
    Returns a dictionary (key identifies an image), of dictionaries of this form:

    {   'img': encode_txt,
        'image_graph': text_graph,
        'txt': encode_txt,
        'text_graph': text_graph}
    :return:
    """

    conf = get_config()
    _, _, test = get_ade20k_split(path_prefix="images")
    img_graphs = json.load(open(conf["scene_graph"]["ade20k_image_sg_test"]))
    txt_graphs = generate_text_graph(test, conf["scene_graph"]["ade20k_text_sg_test"])

    txt_keys = list(txt_graphs.keys())
    for k in list(img_graphs.keys()):
        assert k in txt_keys

    for k in txt_keys:
        item = img_graphs[k]
        if len(item["img"]['entities']) < 2 \
                or len(txt_graphs[k]["txt"]['entities']) < 2 \
                or len(item["img"]['relations']) < 1 \
                or len(txt_graphs[k]["txt"]['relations']) < 1:
            print("no relationship detected, skipping:", k)
            del(img_graphs[k])
            del (txt_graphs[k])
            continue
        else:
            item.update(txt_graphs[k])

    return img_graphs

def get_preprocessed_image_graphs_for_map_world():

    conf = get_config()
    img_graphs = json.load(open(conf["scene_graph"]["ade20k_map_world_preprocessed_img_graph"]))
    return img_graphs

if __name__ == "__main__":
    print("Start")
    conf = get_config()
    train, dev, test = get_ade20k_split(path_prefix="images")
    print(f"Train Split: {len(train)}")
    print(f"Dev Split: {len(dev)}")
    print(f"Test Split: {len(test)}")

    # output_split_list_with_new_prefix(test, "/media/rafi/Samsung_T5/_DATASETS/ADE20K/",
    #                                   "/data/ImageCorpora/ADE20K_2016_07_26/",
    #                                   get_config()["output_dir"] + "/ade20k_caption_test.json")

    graph = get_preprocessed_image_text_graphs_for_test()

    print("Done")
