import os
import json
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def load_tvqa_train_data():
    # base_path = "/net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/text_data/bbox_refined_ts_data"
    # train_path = os.path.join(base_path, "tvqa_bbt_train_bbox_refined_ts_processed_noun_att_labels.json")
    base_path = "/net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/text_data/"
    train_path = os.path.join(base_path, "train_tvshow_v6_srt_ts_fps3_bert_pre_input.json")
    return load_json(train_path)


def mk_bert_lm_input(outfile_path, sub_only=True):
    train_data = load_tvqa_train_data()
    sub_text = {}
    for e in train_data:
        vid_name = e["vid_name"]
        if vid_name not in sub_text:
            sub_text[vid_name] = e["s_tokenized_sub_text"].replace("UNKNAME", "##name").lower()
    lines = []
    empty_line = ""
    for doc in tqdm(sub_text.values()):
        sentences = doc.split(" <eos> ")
        lines.extend(sentences)
        lines.append(empty_line)

    if not sub_only:
        for e in train_data:
            question = e["s_tokenized_q"].lower()
            correct_answer = e["s_tokenized_a"+str(e["answer_idx"])].lower()
            lines.extend([question, correct_answer])
            lines.append(empty_line)

    with open(outfile_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    filepath = "./samples/tvqa_all_sub.txt"
    mk_bert_lm_input(filepath, sub_only=True)

    filepath = "./samples/tvqa_all_sub_qa.txt"
    mk_bert_lm_input(filepath, sub_only=False)
