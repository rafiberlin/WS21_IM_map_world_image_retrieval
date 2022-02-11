from avatar_sgg.dataset.ade20k import get_ade20k_split
from avatar_sgg.config.util import get_config
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_average_similarity_against_generated_caption, \
    compute_recall_on_category, compute_recall_johnson_feiefei, add_inferred_captions, merge_human_captions, \
    use_merged_sequence, run_evaluation

import numpy as np
import os
from avatar_sgg.dataset.ade20k import get_preprocessed_image_text_graphs_for_test
from avatar_sgg.image_retrieval.scene_graph_similarity_model import get_scene_graph_encoder
import torch



if __name__ == "__main__":
    print("Start")
    output_dir = os.path.join(get_config()["output_dir"], "image_retrieval")



    current = get_preprocessed_image_text_graphs_for_test()
    threshold_list = [None]
    # This range has been chosen because the mean of the diagonal on the dev set was around 0.6X
    threshold_list.extend(np.linspace(0.55, 0.7, 15))

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"
    ade20k_category_recall = "ade20k_category_recall"
    fei_fei_recall = "feifei_johnson_recall"
    model = get_scene_graph_encoder()
    model.eval()
    test_results = []

    i = "images/validation/c/childs_room/ADE_val_00000246.jpg"

    with torch.no_grad():
        for k, graph_dict in current.items():

            if len(graph_dict["img"]['entities']) < 2 \
                    or len(graph_dict["txt"]['entities']) < 2 \
                    or len(graph_dict["img"]['relations']) < 1 \
                    or len(graph_dict["txt"]['relations']) < 1 :
                print("relationship detected, skipping:", k)
                continue
            else:
                res = model(graph_dict)
                test_results.append(res)

    results = torch.stack(test_results)
    print("Done")
