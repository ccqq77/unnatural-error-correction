import random
import re
import json
import pandas as pd
import glob
import requests
from datetime import datetime
import argparse
import os
from process import *

def main():
    args = parse_arguments()
    decoder = json.JSONDecoder()

    if args.dataset == "realtimeQA":
        # download RealtimeQA data from https://github.com/realtimeqa/realtimeqa_public
        url = "https://api.github.com/repos/realtimeqa/realtimeqa_public/contents/past/2023/"

        def is_date_in_range(date_str):
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d")
                return start_date <= file_date <= end_date
            except ValueError:
                return False
            
        # use data from 2023/03/17 to 2023/08/04
        start_date = datetime.strptime("20230317", "%Y%m%d")
        end_date = datetime.strptime("20230804", "%Y%m%d")    
        date_pattern = re.compile(r".*(\d{8}).*qa.jsonl")

        response = requests.get(url)
        response.raise_for_status()

        files = []
        for file in response.json():
            match = date_pattern.match(file["name"])
            if match and is_date_in_range(match.group(1)):
                files.append(file)

        for file in files:
            raw_url = file["download_url"]
            file_response = requests.get(raw_url)
            file_response.raise_for_status()

            filename = "dataset/raw/realtimeqa/" + file["name"]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                f.write(file_response.content)

        evidence = {}
        evidence["original"] = []
        question = []
        choice = []
        answer = []
        for filename in glob.glob("dataset/raw/realtimeqa/*qa.jsonl"):
            with open(filename) as f:
                lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                if json_res["evidence"] != "":
                    evidence["original"].append(strip_tags(json_res["evidence"]))
                    question.append(json_res["question_sentence"])
                    choice.append(json_res["choices"])
                    answer.append(json_res["answer"])

        # delete a sample with a duplicate and wrong evidence
        remove_id = question.index("By what percentage have incidents of shoplifting risen in the year to September, according to the Office for National Statistics?")
        del evidence["original"][remove_id], question[remove_id], choice[remove_id], answer[remove_id]
        # sort samples and fix some minor errors
        sorted_pair = sorted(zip(evidence["original"], question, choice, answer))
        evidence["original"], question, choice, answer = zip(*sorted_pair)
        evidence["original"], question, choice, answer = list(evidence["original"]), list(question), list(choice), list(answer)
        evidence["original"][-3] = "T" + evidence["original"][-3]
        evidence["original"] = [scramble_sentence_percent(i, 0) for i in evidence["original"]]
        question = [scramble_sentence_percent(i, 0) for i in question]
        sorted_pair = sorted(zip(evidence["original"], question, choice, answer))
        sorted_pair = list(dict((str(x), x) for x in sorted_pair).values())
        evidence["original"], question, choice, answer = zip(*sorted_pair)
        evidence["original"], question, choice, answer = list(evidence["original"]), list(question), list(choice), list(answer)
        evidence["original"][262] = evidence["original"][262].replace("..", ". ")
        evidence["original"] = [i + "." if i[-1] != "." else i for i in evidence["original"]]
        
        scrambled_sentence = evidence["original"]
        scrambled_id = [[] for _ in range(len(scrambled_sentence))]
        # randomly scramble (stepwise, 1 step = 10%)
        for j in range(10):
            random.seed(args.random_seed)
            scrambled = [scramble_sentence_percent_step(data, (j+1)*0.1, scrambled_id[i]) for i, data in enumerate(scrambled_sentence)]
            scrambled_id = [i[1] for i in scrambled]
            scrambled_sentence = [i[0] for i in scrambled]
            evidence["scrambled_{}%".format((j+1)*10)] = scrambled_sentence

        random.seed(args.random_seed)
        evidence["scrambled_keepfirst"] = [scramble_sentence_keepfirst(i) for i in evidence["original"]]
        random.seed(args.random_seed)
        evidence["scrambled_keepfirstlast"] = [scramble_sentence_keepfirstlast(i) for i in evidence["original"]]
        random.seed(args.random_seed)
        evidence["substituted"] = [substitute_sentence_percent(i, 1) for i in evidence["original"]]

        filename = "dataset/scrambled/realtimeqa/rec.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as wp:
            line = {}
            for j in range(len(evidence["original"])):
                line["original"] = evidence["original"][j]
                line["scrambled_20%"] = evidence["scrambled_20%"][j]
                line["scrambled_50%"] = evidence["scrambled_50%"][j]
                line["scrambled_100%"] = evidence["scrambled_100%"][j]
                line["scrambled_keepfirst"] = evidence["scrambled_keepfirst"][j]
                line["scrambled_keepfirstlast"] = evidence["scrambled_keepfirstlast"][j]
                line["substituted"] = evidence["substituted"][j]
                output_json = json.dumps(line)
                wp.write(output_json + "\n")

        # remove the samples when the provided evidence does not provide sufficient information to answer the question
        remove_lst = [6, 7, 19, 21, 31, 34, 54, 56, 63, 64, 96, 102, 119, 127, 133, 135, 136, 140, 145, 162, 172, 177, 182, 191, 198, 203, 204, 209, 218, 219, 223, 249, 251, 260, 283, 293, 297, 305, 311, 323, 324, 327, 333, 337, 338, 344, 348, 356, 357, 358, 360, 361, 362, 363, 364, 366, 368, 372, 379, 387, 390, 404]

        filename = "dataset/scrambled/realtimeqa/qa.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as wp:
            line = {}
            for j in range(len(evidence["original"])):
                if j not in remove_lst:
                    line["question"] = question[j]
                    line["choice"] = choice[j]
                    line["answer"] = ["A", "B", "C", "D"][int(answer[j][0])]
                    line["evidence_original"] = evidence["original"][j]
                    line["evidence_scrambled_20%"] = evidence["scrambled_20%"][j]
                    line["evidence_scrambled_50%"] = evidence["scrambled_50%"][j]
                    line["evidence_scrambled_100%"] = evidence["scrambled_100%"][j]
                    line["evidence_scrambled_keepfirst"] = evidence["scrambled_keepfirst"][j]
                    line["evidence_scrambled_keepfirstlast"] = evidence["scrambled_keepfirstlast"][j]
                    line["evidence_substituted"] = evidence["substituted"][j]
                    output_json = json.dumps(line)
                    wp.write(output_json + "\n")

    elif args.dataset == "DREAM":
        # download DREAM data from https://github.com/nlpdata/dream
        url_data = "https://api.github.com/repos/nlpdata/dream/contents/data/"
        response = requests.get(url_data)
        response.raise_for_status()
        files = [file for file in response.json() if file["name"] == "dev.json" or file["name"] == "test.json"]

        for file in files:
            raw_url = file["download_url"]
            file_response = requests.get(raw_url)
            file_response.raise_for_status()

            filename = "dataset/raw/dream/" + file["name"]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                f.write(file_response.content)

        # download annotations from https://github.com/nlpdata/dream
        url_anno = "https://api.github.com/repos/nlpdata/dream/contents/annotation/"
        response = requests.get(url_anno)
        response.raise_for_status()
        files = [file for file in response.json()]

        for file in files:
            raw_url = file["download_url"]
            file_response = requests.get(raw_url)
            file_response.raise_for_status()

            filename = "dataset/raw/dream/annotation/" + file["name"]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                f.write(file_response.content)

        with open("dataset/raw/dream/test.json", "r") as f:
            data = json.load(f)
        anno = pd.read_csv("dataset/raw/dream/annotation/annotator1_test.txt", sep = "\t")
        id_lst = [i[2] for i in data]

        dialogue = {}
        dialogue["original"] = []
        question = []
        choice = []
        answer = []
        for j in range(len(anno)):
            id_slc = id_lst.index(anno.dialogueID[j])
            d = "\n".join(data[id_slc][0])
            q = data[id_slc][1][anno.questionIndex[j]-1]["question"]
            c = data[id_slc][1][anno.questionIndex[j]-1]["choice"]
            a_t = data[id_slc][1][anno.questionIndex[j]-1]["answer"]
            a = ["A", "B", "C"][c.index(a_t)]
            
            dialogue["original"].append(d)
            question.append(q)
            choice.append(c)
            answer.append(a)
        
        scrambled_sentence = dialogue["original"]
        scrambled_id = [[] for _ in range(len(scrambled_sentence))]
        # randomly scramble (stepwise, 1 step = 10%)
        for j in range(10):
            random.seed(args.random_seed)
            scrambled = [scramble_sentence_percent_step(data, (j+1)*0.1, scrambled_id[i]) for i, data in enumerate(scrambled_sentence)]
            scrambled_id = [i[1] for i in scrambled]
            scrambled_sentence = [i[0] for i in scrambled]
            dialogue["scrambled_{}%".format((j+1)*10)] = scrambled_sentence
        
        random.seed(args.random_seed)
        dialogue["scrambled_keepfirst"] = [scramble_sentence_keepfirst(i) for i in dialogue["original"]]
        random.seed(args.random_seed)
        dialogue["scrambled_keepfirstlast"] = [scramble_sentence_keepfirstlast(i) for i in dialogue["original"]]
        random.seed(args.random_seed)
        dialogue["substituted"] = [substitute_sentence_percent(i, 1) for i in dialogue["original"]]

        filename = "dataset/scrambled/dream/test.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as wp:
            line = {}
            for j in range(len(dialogue["original"])):
                line["question"] = question[j]
                line["choice"] = choice[j]
                line["answer"] = answer[j]
                line["dialogue_original"] = dialogue["original"][j]
                line["dialogue_scrambled_20%"] = dialogue["scrambled_20%"][j]
                line["dialogue_scrambled_50%"] = dialogue["scrambled_50%"][j]
                line["dialogue_scrambled_100%"] = dialogue["scrambled_100%"][j]
                line["dialogue_scrambled_keepfirst"] = dialogue["scrambled_keepfirst"][j]
                line["dialogue_scrambled_keepfirstlast"] = dialogue["scrambled_keepfirstlast"][j]
                line["dialogue_substituted"] = dialogue["substituted"][j]
                output_json = json.dumps(line)
                wp.write(output_json + "\n")
        
        with open("dataset/raw/dream/dev.json", "r") as f:
            data = json.load(f)
        anno = pd.read_csv("dataset/raw/dream/annotation/annotator2_dev.txt", sep = "\t")
        id_lst = [i[2] for i in data]

        dialogue = {}
        dialogue["original"] = []
        question = []
        choice = []
        answer = []
        for j in range(len(anno)):
            id_slc = id_lst.index(anno.dialogueID[j])
            d = "\n".join(data[id_slc][0])
            q = data[id_slc][1][anno.questionIndex[j]-1]["question"]
            c = data[id_slc][1][anno.questionIndex[j]-1]["choice"]
            a_t = data[id_slc][1][anno.questionIndex[j]-1]["answer"]
            a = ["A", "B", "C"][c.index(a_t)]
            
            dialogue["original"].append(d)
            question.append(q)
            choice.append(c)
            answer.append(a)
        
        scrambled_sentence = dialogue["original"]
        scrambled_id = [[] for _ in range(len(scrambled_sentence))]
        # randomly scramble (stepwise, 1 step = 10%)
        for j in range(10):
            random.seed(args.random_seed)
            scrambled = [scramble_sentence_percent_step(data, (j+1)*0.1, scrambled_id[i]) for i, data in enumerate(scrambled_sentence)]
            scrambled_id = [i[1] for i in scrambled]
            scrambled_sentence = [i[0] for i in scrambled]
            dialogue["scrambled_{}%".format((j+1)*10)] = scrambled_sentence
        
        random.seed(args.random_seed)
        dialogue["scrambled_keepfirst"] = [scramble_sentence_keepfirst(i) for i in dialogue["original"]]
        random.seed(args.random_seed)
        dialogue["scrambled_keepfirstlast"] = [scramble_sentence_keepfirstlast(i) for i in dialogue["original"]]
        random.seed(args.random_seed)
        dialogue["substituted"] = [substitute_sentence_percent(i, 1) for i in dialogue["original"]]

        filename = "dataset/scrambled/dream/dev.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as wp:
            line = {}
            for j in range(len(dialogue["original"])):
                # remove a sample without the annotation from annotator1
                if j != 371:
                    line["question"] = question[j]
                    line["choice"] = choice[j]
                    line["answer"] = answer[j]
                    line["dialogue_original"] = dialogue["original"][j]
                    line["dialogue_scrambled_20%"] = dialogue["scrambled_20%"][j]
                    line["dialogue_scrambled_50%"] = dialogue["scrambled_50%"][j]
                    line["dialogue_scrambled_100%"] = dialogue["scrambled_100%"][j]
                    line["dialogue_scrambled_keepfirst"] = dialogue["scrambled_keepfirst"][j]
                    line["dialogue_scrambled_keepfirstlast"] = dialogue["scrambled_keepfirstlast"][j]
                    line["dialogue_substituted"] = dialogue["substituted"][j]
                    output_json = json.dumps(line)
                    wp.write(output_json + "\n")

    elif args.dataset == "AQuA":
        # download data from https://github.com/google-deepmind/AQuA
        url = "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json"
        
        file_response = requests.get(url)
        file_response.raise_for_status()

        filename = "dataset/raw/aqua/test.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            f.write(file_response.content)
        
        question = {}
        question["original"] = []
        choice = []
        answer = []

        with open("dataset/raw/aqua/test.json") as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                question["original"].append(json_res["question"])
                choice.append(json_res["options"])
                answer.append(json_res["correct"])

        scrambled_sentence = question["original"]
        scrambled_id = [[] for _ in range(len(scrambled_sentence))]
        # randomly scramble (stepwise, 1 step = 10%)
        for j in range(10):
            random.seed(args.random_seed)
            scrambled = [scramble_sentence_percent_step(data, (j+1)*0.1, scrambled_id[i]) for i, data in enumerate(scrambled_sentence)]
            scrambled_id = [i[1] for i in scrambled]
            scrambled_sentence = [i[0] for i in scrambled]
            question["scrambled_{}%".format((j+1)*10)] = scrambled_sentence

        choice = [[m[2:].strip() for m in n] for n in choice]
        choice[29] = [m[2:].strip() for m in choice[29]]

        filename = "dataset/scrambled/aqua/test.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as wp:
            line = {}
            for j in range(len(question["original"])):
                line["choice"] = choice[j]
                line["answer"] = answer[j]
                line["question_original"] = question["original"][j]
                line["question_scrambled_20%"] = question["scrambled_20%"][j]
                line["question_scrambled_50%"] = question["scrambled_50%"][j]
                line["question_scrambled_100%"] = question["scrambled_100%"][j]
                output_json = json.dumps(line)
                wp.write(output_json + "\n")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=10, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="realtimeQA",
        choices=["realtimeQA", "DREAM", "AQuA"]
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()

