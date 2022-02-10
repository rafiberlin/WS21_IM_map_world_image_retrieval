from avatar_sgg.dataset.ade20k import get_ade20k_split
from avatar_sgg.config.util import get_config
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_average_similarity_against_generated_caption, \
    compute_recall_on_category, compute_recall_johnson_feiefei, add_inferred_captions, merge_human_captions, \
    use_merged_sequence, run_evaluation
import numpy as np
import os




if __name__ == "__main__":
    print("Start")
    output_dir = os.path.join(get_config()["output_dir"], "image_retrieval")

    train, dev, test = get_ade20k_split()

    current = test
    threshold_list = [None]
    # This range has been chosen because the mean of the diagonal on the dev set was around 0.6X
    threshold_list.extend(np.linspace(0.55, 0.7, 15))

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"
    ade20k_category_recall = "ade20k_category_recall"
    fei_fei_recall = "feifei_johnson_recall"

    human_caption = "human_captions_query"
    catr_caption = "catr_captions_query"
    merged_human_caption = "merged_human_caption_catr_captions_query"
    merged_sequences_captions = "merged_sequences_catr_captions_query"


    evaluation_name = eval_name(human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(human_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category, output_dir)

    current_inferred_captions = add_inferred_captions(current)
    evaluation_name = eval_name(catr_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current_inferred_captions, compute_average_similarity_against_generated_caption, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(catr_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current_inferred_captions, compute_average_similarity_against_generated_caption, threshold_list, compute_recall_on_category,
                   output_dir)

    current_merged_human = merge_human_captions(current)
    evaluation_name = eval_name(merged_human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current_merged_human, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(merged_human_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current_merged_human, compute_similarity, threshold_list, compute_recall_on_category, output_dir)


    current_merged_sequence = use_merged_sequence(current)
    evaluation_name = eval_name(merged_sequences_captions, fei_fei_recall)
    run_evaluation(evaluation_name, current_merged_sequence, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(merged_sequences_captions, ade20k_category_recall)
    run_evaluation(evaluation_name, current_merged_sequence, compute_similarity, threshold_list, compute_recall_on_category, output_dir)


    print("Done")
