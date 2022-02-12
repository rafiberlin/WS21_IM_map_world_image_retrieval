from avatar_sgg.config.util import get_config
import json
import torch
import torch.utils.data as data
import os

class SceneGraphDataset(data.Dataset):
    """ SGEncoding dataset """

    def __init__(self, train_ids, test_ids, test_on=False, val_on=False, num_test=5000, num_val=5000):
        super(SceneGraphDataset, self).__init__()

        conf = get_config()["scene_graph"]
        cap_graph_file = conf["capgraphs_file"]
        vg_dict_file = conf["visual_genome_dict_file"]
        image_data_file = conf["image_data"]
        image_dir = conf["image_dir"]

        self.cap_graph = json.load(open(cap_graph_file))
        self.image_data = json.load(open(image_data_file))


        self.coco_ids_to_image_path = {
            str(self.image_data[i]["coco_id"]): os.path.join(image_dir, str(self.image_data[i]["image_id"]) + ".jpg") for
        i, entry
            in enumerate(self.cap_graph["vg_coco_ids"]) if entry > -1}

        # self.coco_image_data = [str(self.image_data[i]) for i, entry in enumerate(self.cap_graph["vg_coco_ids"]) if
        #                         entry > -1]
        # Warning somehow, not the same size...
        # num_good_paths = sum([1 for entry in self.coco_image_data if
        #      os.path.exists(image_dir + str(entry["image_id"]) + ".jpg")])
        # num_coco_ids = len(cap_graph['vg_coco_id_to_caps'])
        # assert num_good_paths == num_coco_ids

        self.train_ids = train_ids
        self.test_ids = test_ids
        if test_on:
            self.key_list = self.test_ids[:num_test]
        elif val_on:
            self.key_list = self.test_ids[num_test:num_test + num_val]
        else:
            self.key_list = self.test_ids[num_test + num_val:] + self.train_ids

    def __getitem__(self, item):

        coco_id = self.key_list[item]
        return self.coco_ids_to_image_path[coco_id], self.cap_graph["vg_coco_id_to_caps"][coco_id]

    def __len__(self):
        return len(self.key_list)


class SimpleCollator(object):
    def __call__(self, batch):

        # Just use 5 captions to ease the use for tensors later on
        glue = {path:{"caption": captions[:5]} for path, captions in batch}

        return glue


def get_scene_graph_splits():
    """
    Get the training split used for Image Retrieval using Scence Graph
    from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/
    :return: a tuple with training and test ids (intersection between Visual Genome and MSCOCO), and the linked data
    """
    conf = get_config()
    output_dir = conf["output_dir"]
    conf = conf["scene_graph"]

    train_id_file = os.path.join(output_dir, 'train_ids.json')
    test_id_file = os.path.join(output_dir, 'test_ids.json')

    ids_created = os.path.exists(train_id_file)

    if not ids_created:

        sg_train_path = conf["train"]
        sg_test_path = conf["test"]
        sg_val_path = conf["val"]
        print("Loading samples. This can take a while.")
        sg_data_train = json.load(open(sg_train_path))
        sg_data_val = json.load(open(sg_val_path))
        sg_data_test = json.load(open(sg_test_path))
        # sg_data = torch.load(sg_train_path)
        # sg_data.update(torch.load(sg_test_path))
        # Merge the val sample to the training data, it would be a waste...
        sg_data_train.update(sg_data_val)
        sg_data = sg_data_train.copy()
        sg_data.update(sg_data_test)
        train_ids = list(sg_data_train.keys())
        test_ids = list(sg_data_test.keys())

        with open(train_id_file, 'w') as f:
            json.dump(train_ids, f)
        print("created:", train_id_file)
        with open(test_id_file, 'w') as f:
            json.dump(test_ids, f)
        print("created:", test_id_file)
    else:
        train_ids = json.load(open(train_id_file))
        test_ids = json.load(open(test_id_file))

    print("Number of Training Samples", len(train_ids))
    print("Number of Testing Samples", len(test_ids))
    return train_ids, test_ids


def get_scene_graph_loader(batch_size, train_ids, test_ids, test_on=False, val_on=False, num_test=5000, num_val=1000):
    """ Returns a data loader for the desired split """
    split = SceneGraphDataset(train_ids, test_ids, test_on=test_on, val_on=val_on, num_test=num_test,
                              num_val=num_val)

    loader = torch.utils.data.DataLoader(split,
                                         batch_size=batch_size,
                                         shuffle=not (test_on or val_on),  # only shuffle the data in training
                                         pin_memory=True,
                                         # num_workers=4,
                                         collate_fn=SimpleCollator(),
                                         )
    return loader

if __name__ == "__main__":
    print("Start")
    batch_size = 100
    train_ids, test_ids = get_scene_graph_splits()

    use_test = True
    use_val = False
    loader = get_scene_graph_loader(batch_size, train_ids, test_ids, test_on=use_test, val_on=use_val)
    next(enumerate(loader))
    print("Done")
