from avatar_sgg.dataset.ade20k import get_ade20k_split
from avatar_sgg.config.util import get_config
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_scene_graph_similarity, \
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

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"
    ade20k_category_recall = "ade20k_category_recall"
    fei_fei_recall = "feifei_johnson_recall"
    text_scene_graph_query = "text_scene_graph_query "
    evaluation_name = eval_name(text_scene_graph_query , fei_fei_recall)
    threshold_list.extend(np.linspace(0.65, 0.85, 15))
    run_evaluation(evaluation_name, current, compute_scene_graph_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(text_scene_graph_query, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_scene_graph_similarity, threshold_list, compute_recall_on_category,
                   output_dir)

    print("Done")
