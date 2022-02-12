from avatar_sgg.dataset.visual_genome import get_scene_graph_splits, get_scene_graph_loader
from avatar_sgg.config.util import get_config
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_average_similarity_against_generated_caption, \
    compute_recall_johnson_feiefei, add_inferred_captions, merge_human_captions, \
    run_evaluation
import argparse
import numpy as np
import os




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Retrieval on Visual Genome(MS COCO images) - SentenceBert Similarity")
    parser.add_argument(
        "--gallery_size",
        default=50,
        metavar="",
        help="Size of the gallery for image retrieval",
        type=str,
    )

    args = parser.parse_args()

    print("Start")
    output_dir = os.path.join(get_config()["output_dir"], "image_retrieval")

    gallery_size = args.gallery_size
    train_ids, test_ids = get_scene_graph_splits()

    use_test = True
    use_val = False
    loader = get_scene_graph_loader(gallery_size, train_ids, test_ids, test_on=use_test, val_on=use_val)
    _ , current = next(enumerate(loader))


    threshold_list = [None]
    # This range has been chosen because the mean of the diagonal on the dev set was around 0.6X
    threshold_list.extend(np.linspace(0.55, 0.7, 15))

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"
    human_caption = f"vg_{gallery_size}_human_captions_query"
    catr_caption = f"vg_{gallery_size}_catr_captions_query"
    merged_human_caption = f"vg_{gallery_size}_merged_human_caption_catr_captions_query"
    fei_fei_recall = "feifei_johnson_recall"

    evaluation_name = eval_name(human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)


    current_inferred = add_inferred_captions(current)
    evaluation_name = eval_name(catr_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current_inferred, compute_average_similarity_against_generated_caption, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)


    current_merged = merge_human_captions(current)
    evaluation_name = eval_name(merged_human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current_merged, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    print("Done")
