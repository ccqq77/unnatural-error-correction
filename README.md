# Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Scrambled Text

Code for the paper [Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Scrambled Text](https://arxiv.org/abs/2311.18805).

```bibtex
@inproceedings{cao2023unnatural,
  title={Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural Scrambled Text},
  author={Cao, Qi and Kojima, Takeshi and Matsuo, Yutaka and Iwasawa, Yusuke},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={8898--8913},
  year={2023}
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

