import argparse
import os
from utils import *
from process import *


def main():
    args = parse_arguments()
    print("*"*50)

    fix_seed(args.random_seed)

    decoder = Decoder(args)
    dataloader = setup_data_loader(args)

    print("model: " + args.model)
    print("method: " + args.method)
    if args.method == "few-shot-cot":
        print("demo_scramble: " + args.demo_question_scramble_rate)
    print("task: " + args.task)
    print("scramble: " + args.scramble)
    print_now()

    if args.method == "few-shot-cot":
        filename = args.output_dir + "/" + args.task + "/" + args.dataset + "/" + args.method + "/" + "demo_random_" + args.demo_question_scramble_rate + "/" + args.scramble + "/" + args.model
    else:
        filename = args.output_dir + "/" + args.task + "/" + args.dataset + "/" + args.method + "/" + args.scramble + "/" + args.model
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if args.task == "scrambled_rec":
        with open(filename, "w") as wp:
            dis_list = []
            for i, data in enumerate(dataloader):
                x, y = data
                output_line = {}
                output_line["scrambled"] = x
                output_line["original"] = y

                x = [j.strip() for j in x]
                y = [j.strip() for j in y]

                if args.method == "zero-shot":
                    instruction = "The following sentence contains words with scrambled letters. Please recover the original sentence from it."
                    x = [instruction + "\n" + j for j in x]

                elif args.method == "few-shot":
                    # samples from wikiQA dataset https://huggingface.co/datasets/wiki_qa
                    examples = [
                        "The camp continued to function this way until the war ended.",
                        "It was first developed in the 1980s by Acorn Computers Ltd to power their desktop machines and subsequently spun off as a separate company, now ARM Holdings.",
                        "According to the CIA Factbook, the United States is one of three countries (the others being Liberia and Burma/Myanmar) that has not adopted the International System of Units (SI) metric system as their official system of weights and measures.",
                        ]
                    
                    demo = generate_demo_rec(examples, 0.5, args.random_seed)
                    x = [demo + j for j in x]

                else:
                    raise ValueError("method is not properly defined ...")
                
                max_length = args.max_length
                pred_lst = decoder.decode(args, x, max_length, 1, args.temperature)
                pred_lst_clean = [answer_cleansing(args, j) if j != "" else j for j in pred_lst]
                    
                output_line["recovered"] = pred_lst_clean
                output_line["input"] = x
                output_line["output"] = pred_lst
                output_line = [dict(zip(output_line, t)) for t in zip(*output_line.values())]
                for j in output_line:
                    output_json = json.dumps(j)
                    wp.write(output_json + "\n")
                
                for j in range(len(x)):
                    print("*"*50)
                    print("No.{}".format(i * args.batch_size + j + 1))
                    print("#Input:\n" + x[j])
                    print("#Output:\n" + pred_lst[j])
                    print("#Pred:\n{}".format(pred_lst_clean[j]))
                    print("#GT:\n" + y[j])
                    print("*"*50)

                    dis = edit_distance(pred_lst_clean[j], y[j])
                    dis_list.append(dis)

                if (args.limit_dataset_size != 0) and ((i + 1) * args.batch_size >= args.limit_dataset_size):
                    break
                    # raise ValueError("Stop !!")

            accuracy = np.mean(dis_list)
            print("edit_dis : {}".format(accuracy))

    elif args.task == "scrambled_qa":
        with open(filename, "w") as wp:
            correct_list = []
            total = 0

            for i, data in enumerate(dataloader):
                
                if len(data) == 3:
                    x, y, choice = data
                else:
                    x, y = data
                output_line = {}
                
                x = [j.strip() for j in x]
                if args.dataset == "scrambled_AQuA":
                    examples = [
                        {
                            "question": "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?",
                            "choice": "(A)50 (B)45 (C)65 (D)78 (E)64",
                            "answer": "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. The answer is (A)."
                        },
                        {
                            "question": "If a / b = 3/4 and 8a + 5b = 22, then find the value of a.",
                            "choice": "(A)1/2 (B)3/2 (C)5/2 (D)4/2 (E)7/2",
                            "answer": "If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. The answer is (B)."
                        },
                        {
                            "question": "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?",
                            "choice": "(A)53 km (B)55 km (C)52 km (D)60 km (E)50 km",
                            "answer": "The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (E).",
                        },
                        {
                            "question": "How many keystrokes are needed to type the numbers from 1 to 500?",
                            "choice": "(A)1156 (B)1392 (C)1480 (D)1562 (E)1788",
                            "answer": "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. The answer is (B)."
                        },
                    ]
                    
                    if args.method == "few-shot-cot":
                        demo_question_scramble_rate = float(args.demo_question_scramble_rate.strip("%"))/100
                        demo = generate_demo_aqua(examples, demo_question_scramble_rate, args.random_seed)
                        x = [demo + j for j in x]
                    else:
                        raise ValueError("method is not properly defined ...")
                    
                else:
                    if args.method == "zero-shot":
                        pass
                    else:
                        raise ValueError("method is not properly defined ...")


                output_line["input"] = x

                max_length = args.max_length

                pred_lst = decoder.decode(args, x, max_length, 1, args.temperature)
                pred_lst_clean = [answer_cleansing(args, pred_lst[j], choice[j]) if pred_lst[j] != "" else pred_lst[j] for j in range(len(pred_lst))]
                output_line["output"] = pred_lst
                output_line["pred"] = pred_lst_clean
                output_line["GT"] = y
                
                output_line = [dict(zip(output_line, t)) for t in zip(*output_line.values())]
                for j in output_line:
                    output_json = json.dumps(j)
                    wp.write(output_json + "\n")

                for j in range(len(x)):
                    print("*"*50)
                    print("No.{}".format(i * args.batch_size + j + 1))
                    print("#Input:\n" + x[j])
                    print("#Output:\n" + pred_lst[j])
                    print("#Pred:\n{}".format(pred_lst_clean[j]))
                    print("#GT:\n" + y[j])
                    print("*"*50)

                    correct = compare_answer(pred_lst_clean[j], y[j])
                    correct_list.append(correct)
                    total += 1

                if (args.limit_dataset_size != 0) and ((i + 1) * args.batch_size >= args.limit_dataset_size):
                    break

            accuracy = (sum(correct_list) * 1.0 / total) * 100
            print("Accuracy : {}".format(accuracy))
            
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=10)

    parser.add_argument("--task", type=str, default="scrambled_rec", choices=["scrambled_qa", "scrambled_rec"])

    parser.add_argument("--scramble", type=str, default="random_100%", choices=["original", "random_20%", "random_50%", "random_100%", "keepfirst", "keepfirstlast", "substituted"])

    parser.add_argument("--api_key", type=str, default=None)

    parser.add_argument(
        "--dataset", type=str, default="scrambled_realtimeQA",
        choices=["scrambled_realtimeQA", "scrambled_DREAM_test", "scrambled_DREAM_dev", "scrambled_AQuA"]
    )

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--max_num_worker", type=int, default=1, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt-4-0314",
        choices=["text-davinci-003", "gpt-3.5-turbo-0301", "gpt-4-0314",
                 "falcon-180B", "falcon-180B-chat",
                 "falcon-40b", "falcon-40b-instruct",
                 "falcon-7b", "falcon-7b-instruct", 
                 "Llama-2-70b-hf", "Llama-2-70b-chat-hf",
                 "Llama-2-13b-hf", "Llama-2-13b-chat-hf", 
                 "Llama-2-7b-hf", "Llama-2-7b-chat-hf",
                 "mpt-30b", "mpt-30b-instruct",
                 "ul2", "flan-ul2", "flan-t5-xxl", "byt5-xxl",
                 ]
                 )

    parser.add_argument(
        "--method", type=str, default="zero-shot",
        choices=["zero-shot", "few-shot", "few-shot-cot"]
        )
    
    parser.add_argument(
        "--demo_question_scramble_rate", type=str, default="0%",
        choices=["0%", "20%", "50%", "100%"], help="scramble rate of questions in demo for few-shot-cot AQuA"
        )

    parser.add_argument(
        "--max_length", type=int, default=256,
        )

    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
        )

    parser.add_argument(
        "--output_dir", type=str, default="./output", help="output directory"
        )

    parser.add_argument(
        "--temperature", type=float, default=0.0
        )

    args = parser.parse_args()

    if args.dataset in ["scrambled_realtimeQA"]:
        if args.task == "scrambled_qa":
            args.dataset_path = "dataset/scrambled/realtimeqa/qa.json"
        elif args.task == "scrambled_rec":
            args.dataset_path = "dataset/scrambled/realtimeqa/rec.json"
    elif args.dataset in ["scrambled_AQuA"]:
        if args.task == "scrambled_qa":
            args.dataset_path = "dataset/scrambled/aqua/test.json"
    elif args.dataset in ["scrambled_DREAM_test"]:
        if args.task == "scrambled_qa":
            args.dataset_path = "dataset/scrambled/dream/test.json"
    elif args.dataset in ["scrambled_DREAM_dev"]:
        if args.task == "scrambled_qa":
            args.dataset_path = "dataset/scrambled/dream/dev.json"
    else:
        raise ValueError("dataset is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
