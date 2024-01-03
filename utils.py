"""
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
"""

from torch.utils.data import Dataset
import openai
import multiprocessing
import json
import torch
import random
import torch
import datetime
import json
from transformers import AutoTokenizer, AutoModelForCausalLM , T5ForConditionalGeneration, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import re
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import BitsAndBytesConfig

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    now = now.strftime("%Y/%m/%d %H:%M:%S")
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

def decoder_for_hf(args, input, max_length, n, t, model, tokenizer):
    if "Llama-2" in args.model:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
        
        if t == 0:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=False,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [13, tokenizer.eos_token_id],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask
            )
        else:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=t,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [13, tokenizer.eos_token_id],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask
            )

        prompt_length = [len(tokenizer.decode(i, skip_special_tokens=True,)) for i in inputs.input_ids]
        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        result = [result[i][prompt_length[i]:] for i in range(len(result))]

    elif args.model in ["ul2", "flan-ul2", "flan-t5-xxl"]:
        if args.task == "scrambled_rec":
            class KeywordsStoppingCriteria(StoppingCriteria):
                def __init__(self, keywords_ids:list):
                    self.keywords = keywords_ids

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    if list(input_ids[0][-len(self.keywords):]) == self.keywords:
                        return True
                    else:
                        return False

            # prevent repetition
            stop_words = "Scrambled sentence:"
            stop_ids = tokenizer.encode(stop_words)
        
            inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
            
            if t == 0:
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=False,
                    bos_token_id = tokenizer.bos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    attention_mask=inputs.attention_mask,
                    stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop_ids)])
                    )
            else:
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=t,
                    bos_token_id = tokenizer.bos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    attention_mask=inputs.attention_mask,
                    stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop_ids)])
                    )

            result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        
        else:
            inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")

            if t == 0:
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=False,
                    bos_token_id = tokenizer.bos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    attention_mask=inputs.attention_mask,
                    )
            else:
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=t,
                    bos_token_id = tokenizer.bos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    attention_mask=inputs.attention_mask,
                    )

        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    
    elif args.model in ["byt5-xxl"]:
        
        inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
        
        if t == 0:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length*10,
                do_sample=False,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [13, tokenizer.eos_token_id],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )
        else:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length*10,
                do_sample=True,
                temperature=t,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [13, tokenizer.eos_token_id],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )
        
        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

    elif args.model in ["falcon-40b", "falcon-40b-instruct", "falcon-7b", "falcon-7b-instruct"]:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
                
        inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
        
        if t == 0:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=False,
                eos_token_id = [tokenizer.eos_token_id, 193],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )
        else:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=t,
                eos_token_id = [tokenizer.eos_token_id, 193],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )

        prompt_length = [len(tokenizer.decode(i, skip_special_tokens=True,)) for i in inputs.input_ids]
        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        result = [result[i][prompt_length[i]:] for i in range(len(result))]

    elif args.model in ["falcon-180B", "falcon-180B-chat"]:
        tokenizer.pad_token = tokenizer.eos_token
                
        inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
        
        if t == 0:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=False,
                eos_token_id = [tokenizer.eos_token_id, 193],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )
        else:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=t,
                eos_token_id = [tokenizer.eos_token_id, 193],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )

        prompt_length = [len(tokenizer.decode(i, skip_special_tokens=True,)) for i in inputs.input_ids]
        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        result = [result[i][prompt_length[i]:] for i in range(len(result))]

    elif args.model in ["mpt-30b-instruct"]:
        
        inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
        
        if t == 0:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=False,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [tokenizer.eos_token_id, 187],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )
        else:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=t,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [tokenizer.eos_token_id, 187],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )

        prompt_length = [len(tokenizer.decode(i, skip_special_tokens=True,)) for i in inputs.input_ids]
        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        result = [result[i][prompt_length[i]:] for i in range(len(result))]
    
    elif args.model in ["mpt-30b"]:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        inputs = tokenizer(input, return_tensors="pt", padding=True).to("cuda")
        
        if t == 0:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=False,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [tokenizer.eos_token_id, 187],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )
        else:
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=t,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = [tokenizer.eos_token_id, 187],
                pad_token_id = tokenizer.pad_token_id,
                return_dict_in_generate=True,
                attention_mask=inputs.attention_mask,
                )

        prompt_length = [len(tokenizer.decode(i, skip_special_tokens=True,)) for i in inputs.input_ids]
        result = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        result = [result[i][prompt_length[i]:] for i in range(len(result))]
    
    return result

def decoder_for_openai(args, input, max_length, n, t):
    
    openai.api_key = args.api_key
    
    output = []
    for i in input:
        if args.model in ["text-davinci-003"]:
            response = openai.Completion.create(
                engine=args.model,
                prompt=i,
                max_tokens=max_length,
                temperature=t,
                stop=None,
                n=n
                )

            if n == 1:
                output.append(response["choices"][0]["text"])
            else:
                output.append(response)
                
        else:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": i}
                ],
                max_tokens=max_length,
                temperature=t,
                stop=None,
                n=n
            )
            if n == 1:
                output.append(response["choices"][0]["message"]["content"])
            else:
                output.append(response)
        
        
        
    return output

class Decoder():
    def __init__(self, args):
        if args.model in ["falcon-40b", "falcon-40b-instruct", "mpt-30b", "mpt-30b-instruct", "falcon-7b", "falcon-7b-instruct"]:
            model_id = args.model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                           padding_side="left",
                                                           legacy=False
                                                           )
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

        elif args.model in ["falcon-180B", "falcon-180B-chat"]:
            model_id = args.model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", legacy=False)
            if args.dataset in ["scrambled_DREAM_test", "scrambled_DREAM_dev"]:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, torch_dtype="auto")
            else:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
                self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, torch_dtype="auto")

        elif args.model in ["ul2", "flan-ul2", "flan-t5-xxl", "byt5-xxl"]:
            model_id = args.model
            self.model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", legacy=False)

        elif args.model in ["Llama-2-70b-hf", "Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-7b-chat-hf"]:
            model_id = args.model
            if args.model in ["Llama-2-13b-hf", "Llama-2-13b-chat-hf"]:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="balanced_low_0", torch_dtype="auto")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
 
    def decode(self, args, input, max_length, n, t):
        if args.model in ["falcon-180B", "falcon-180B-chat", 
                          "falcon-40b", "falcon-40b-instruct", 
                          "falcon-7b", "falcon-7b-instruct", 
                          "mpt-30b", "mpt-30b-instruct", 
                          "Llama-2-70b-hf", "Llama-2-13b-hf", "Llama-2-7b-hf", 
                          "Llama-2-70b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-7b-chat-hf",
                          "ul2", "flan-ul2", "flan-t5-xxl", "byt5-xxl"]:
            response = decoder_for_hf(args, input, max_length, n, t, self.model, self.tokenizer)
        elif "gpt-4" in args.model or "gpt-3.5" in args.model or "davinci" in args.model:
            response = decoder_for_openai(args, input, max_length, n, t)
        return response

def data_reader(args):

    questions = []
    answers = []

    if args.dataset in ["scrambled_realtimeQA"]:
        if args.task == "scrambled_rec":
            with open(args.dataset_path) as f:
                ground_truth = []
                input_prompt = []
                for line in f.readlines():
                    ground_truth.append(json.loads(line)["original"])
                    if args.scramble == "random_100%":
                        scrambled = json.loads(line)["scrambled_100%"]
                    elif args.scramble == "random_50%":
                        scrambled = json.loads(line)["scrambled_50%"]
                    elif args.scramble == "random_20%":
                        scrambled = json.loads(line)["scrambled_20%"]
                    elif args.scramble == "keepfirst":
                        scrambled = json.loads(line)["scrambled_keepfirst"]
                    elif args.scramble == "keepfirstlast":
                        scrambled = json.loads(line)["scrambled_keepfirstlast"]
                    input_prompt.append("Scrambled sentence: " + scrambled + "\n" + "Recovered sentence:")

        elif args.task == "scrambled_qa":
            with open(args.dataset_path) as f:
                input_prompt = []
                ground_truth = []
                choice = []
                for line in f.readlines():
                    if args.scramble == "original":
                        evidence = json.loads(line)["evidence_original"]
                    elif args.scramble == "random_100%":
                        evidence = json.loads(line)["evidence_scrambled_100%"]
                    elif args.scramble == "random_50%":
                        evidence = json.loads(line)["evidence_scrambled_50%"]
                    elif args.scramble == "random_20%":
                        evidence = json.loads(line)["evidence_scrambled_20%"]
                    elif args.scramble == "keepfirst":
                        evidence = json.loads(line)["evidence_scrambled_keepfirst"]
                    elif args.scramble == "keepfirstlast":
                        evidence = json.loads(line)["evidence_scrambled_keepfirstlast"]
                    elif args.scramble == "substituted":
                        evidence = json.loads(line)["evidence_substituted"]

                    input_prompt.append("Question: " + json.loads(line)["question"] +
                                        "\nChoices: " + "".join(["(" + ["A", "B", "C", "D"][i] + ")" + json.loads(line)["choice"][i] + " " for i in range(len(json.loads(line)["choice"]))]).strip() +
                                        "\nEvidence: " + evidence +
                                        "\nAnswer: " + "Based on the evidence, among A through D, the answer is")
                    ground_truth.append(json.loads(line)["answer"])
                    choice.append(json.loads(line)["choice"])
              
    elif args.dataset in ["scrambled_DREAM_test", "scrambled_DREAM_dev"]:
        if args.task == "scrambled_qa":
            with open(args.dataset_path) as f:
                input_prompt = []
                ground_truth = []
                choice = []
                for line in f.readlines():
                    if args.scramble == "original":
                        dialogue = json.loads(line)["dialogue_original"]
                    elif args.scramble == "random_100%":
                        dialogue = json.loads(line)["dialogue_scrambled_100%"]
                    elif args.scramble == "random_50%":
                        dialogue = json.loads(line)["dialogue_scrambled_50%"]
                    elif args.scramble == "random_20%":
                        dialogue = json.loads(line)["dialogue_scrambled_20%"]
                    elif args.scramble == "keepfirst":
                        dialogue = json.loads(line)["dialogue_scrambled_keepfirst"]
                    elif args.scramble == "keepfirstlast":
                        dialogue = json.loads(line)["dialogue_scrambled_keepfirstlast"]
                    elif args.scramble == "substituted":
                        dialogue = json.loads(line)["dialogue_substituted"]

                    input_prompt.append("Dialogue:\n" + dialogue +
                                        "\nQuestion: " + json.loads(line)["question"] +
                                        "\nChoices: " + "".join(["(" + ["A", "B", "C"][i] + ")" + json.loads(line)["choice"][i] + " " for i in range(len(json.loads(line)["choice"]))]).strip() +
                                        "\nAnswer: " + "Based on the dialogue, among A through C, the answer is")
                    ground_truth.append(json.loads(line)["answer"])
                    choice.append(json.loads(line)["choice"])
                    
    elif args.dataset in ["scrambled_AQuA"]:
        if args.task == "scrambled_qa":
            with open(args.dataset_path) as f:
                input_prompt = []
                ground_truth = []
                choice = []
                for line in f.readlines():
                    if args.scramble == "original":
                        question = json.loads(line)["question_original"]
                    elif args.scramble == "random_100%":
                        question = json.loads(line)["question_scrambled_100%"]
                    elif args.scramble == "random_50%":
                        question = json.loads(line)["question_scrambled_50%"]
                    elif args.scramble == "random_20%":
                        question = json.loads(line)["dialogue_scrambled_20%"]

                    input_prompt.append("Question: " + question +
                                        "\nChoices: " + "".join(["(" + ["A", "B", "C", "D", "E"][i] + ")" + json.loads(line)["choice"][i] + " " for i in range(len(json.loads(line)["choice"]))]).strip() +
                                        "\nAnswer:")
                    ground_truth.append(json.loads(line)["answer"])
                    choice.append(json.loads(line)["choice"])
            
    else:
        raise ValueError("dataset is not properly defined ...")
    
    print("dataset: {}".format(args.dataset))
    print("data_size: {}".format(len(input_prompt)))
    
    if args.task == "scrambled_rec":
        return input_prompt, ground_truth
    elif args.task == "scrambled_qa":
        return input_prompt, ground_truth, choice

class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        if args.task == "scrambled_rec": 
            self.input_prompt, self.ground_truth = data_reader(args)
            self.len = len(self.input_prompt)
        elif args.task == "scrambled_qa":
            self.input_prompt, self.ground_truth, self.choice = data_reader(args)
            self.len = len(self.input_prompt)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        try:
            return self.input_prompt[index], self.ground_truth[index], self.choice[index]
        except:
            return self.input_prompt[index], self.ground_truth[index]


def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    
    dataset = MyDataset(args)
    
    if len(dataset[0]) == 3:
        def collate_fn(items):
            x = [i[0] for i in items]
            y = [i[1] for i in items]
            z = [i[2] for i in items]
            return x, y, z
        
        dataloader = torch.utils.data.DataLoader(dataset,
                    shuffle=False,
                    batch_size=args.batch_size,
                    drop_last=False,
                    num_workers=dataloader_num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                    pin_memory=True,
                    collate_fn=collate_fn)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                    shuffle=False,
                    batch_size=args.batch_size,
                    drop_last=False,
                    num_workers=dataloader_num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                    pin_memory=True)

    return dataloader

def contains_substring(lst, substr):
    return any(substr in str(element) for element in lst)

def answer_cleansing(args, pred, choice=None):
    if choice != None and args.dataset != "scrambled_AQuA":
        if contains_substring(choice, ".") == False:
            pred = pred.split(".")[0]
    if args.dataset in ["scrambled_realtimeQA"]:
        if args.task == "scrambled_qa":
            # remove repitition
            pred = pred.split("Answer: ")[0]
            pred = pred.split("Question: ")[0]
            pred = pred.split("Choices: ")[0]
            pred = pred.split("Evidence: ")[0]
            pred_abcd = re.findall(r"A|B|C|D", pred)
            if choice != None:
                # sometimes, models output the text of choice rather than ABCD
                pred_text = re.findall(r"|".join((re.escape(c) for c in choice)), pred)
                if len(pred_text) != 0:
                    for t1 in pred_text:
                        if any(t1 in t2 for t2 in pred_text if t1 != t2):
                            pred_text.remove(t1)
                    pred_text = [["A", "B", "C", "D"][choice.index(t)] for t in pred_text]
            if len(pred_abcd) == 0 and len(pred_text) == 0:
                pred = ""
            elif len(pred_text) == 0:
                pred = pred_abcd[0]
            else:
                pred = pred_text[0]
        elif args.task == "scrambled_rec":
            if "Scrambled sentence:" in pred:
                pred = pred.split("Scrambled sentence:")[0]
            pred = pred.strip()
            
    elif args.dataset in ["scrambled_DREAM_test", "scrambled_DREAM_dev"]:
        if args.task == "scrambled_qa":
            # remove repitition
            pred = pred.split("Answer: ")[0]
            pred = pred.split("Question: ")[0]
            pred = pred.split("Choices: ")[0]
            pred = pred.split("Dialogue: ")[0]
            pred_abc = re.findall(r"A|B|C", pred)
            if choice != None:
                pred_text = re.findall(r"|".join((re.escape(c) for c in choice)), pred)
                if len(pred_text) != 0:
                    for t1 in pred_text:
                        if any(t1 in t2 for t2 in pred_text if t1 != t2):
                            pred_text.remove(t1)
                    pred_text = [["A", "B", "C"][choice.index(t)] for t in pred_text]
            if len(pred_abc) == 0 and len(pred_text) == 0:
                pred = ""
            elif len(pred_text) == 0:
                pred = pred_abc[0]
            else:
                pred = pred_text[0]

    elif args.dataset in ["scrambled_AQuA"]:
        if args.task == "scrambled_qa":
            if "answer is " in pred:
                pred = pred.split("answer is ")[-1]
                pred = re.findall(r"A|B|C|D|E", pred)
                if len(pred) == 0:
                    pred = ""
                else:
                    pred = pred[0]
            else:
                pred = ""
    else:
        raise ValueError("dataset is not properly defined ...")
    
    return pred

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]

def compare_answer(pred, label):
    try:
        pred = eval(pred)
        label = eval(label)
    except:
        pred = pred
        label = label

    return (pred == label)