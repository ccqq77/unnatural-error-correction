# Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural Scrambled Text

Code for the paper [Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural Scrambled Text](https://arxiv.org/abs/2311.18805).

```bibtex
@inproceedings{cao-etal-2023-unnatural,
    title = "Unnatural Error Correction: {GPT}-4 Can Almost Perfectly Handle Unnatural Scrambled Text",
    author = "Cao, Qi  and
      Kojima, Takeshi  and
      Matsuo, Yutaka  and
      Iwasawa, Yusuke",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.550/",
    doi = "10.18653/v1/2023.emnlp-main.550",
    pages = "8898--8913"
}
```

## Usage
Download the original datasets and create Scrambled Bench:
```
python create.py --dataset realtimeQA
python create.py --dataset DREAM
python create.py --dataset AQuA
```

Run the experiments using different models in different settings, for example:
```
python evaluate.py --task scrambled_rec --dataset scrambled_realtimeQA --method zero-shot --model gpt-4-0314 --api_key YOUR_OPENAI_KEY
```

