import random
import re
import numpy as np
import string
from io import StringIO
from html.parser import HTMLParser
import json


# from https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def shuffle_different(x):
    slc = random.sample(range(1, len(x)), 1)[0]
    for i in reversed(range(1, len(x))):
        if i == slc:
            # pick an element in x[:i] with which to exchange x[i]
            j = int(random.random() * i)
            x[i], x[j] = x[j], x[i]
        else:
            j = int(random.random() * (i+1))
            x[i], x[j] = x[j], x[i]

def scramble_word(word):
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 1:
        return word
    else:
        shuffle_different(letters)
        result = []
        for c in word:
            if c.isalpha():
                result.append(letters.pop(0))
            else:
                result.append(c)
        return "".join(result)
    
def scramble_word_keepfirst(word):
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 2:
        return word
    else:
        letters = letters[1:]
        shuffle_different(letters)
        result = []
        for l in range(len(word)):
            if not word[0].isalpha():
                result.append(word[0])
                word = word[1:]
            else:
                result.append(word[0])
                for c in word[1:]:
                    if c.isalpha():
                        result.append(letters.pop(0))
                    else:
                        result.append(c)
                break
        return "".join(result)

def scramble_word_keepfirstlast(word):
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 3:
        return word
    else:
        letters = letters[1:-1]
        shuffle_different(letters)
        head = []
        tail = []
        for l in range(len(word)):
            if not word[0].isalpha():
                head.append(word[0])
                word = word[1:]
            if not word[-1].isalpha():
                tail.append(word[-1])
                word = word[:-1]
            if word[0].isalpha() and word[-1].isalpha():
                head.append(word[0])
                for c in word[1:-1]:
                    if c.isalpha():
                        head.append(letters.pop(0))
                    else:
                        head.append(c)
                head.append(word[-1])
                break
        result = head + list(reversed(tail))
        return "".join(result)

def substitute_word(word):
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 1:
        return word
    else:
        random_letters = []
        for l in letters:
            random_letters.append(random.choice(string.ascii_letters.replace(l, "")))
        result = []
        for c in word:
            if c.isalpha():
                result.append(random_letters.pop(0))
            else:
                result.append(c)
        return "".join(result)
    
def scramble_sentence_percent(sentence, percent):
    words = re.split("(\W)", sentence)
    words_id = []
    for i in list(range(len(words))):
        letters = [c for c in words[i] if c.isalpha()]
        if len(letters) > 1:
            words_id.append(i)
    words_id_slc = random.sample(words_id, k=round(len(words_id)*percent))
    scrambled_words = words
    for i in words_id_slc:
        scrambled_words[i] = scramble_word(words[i])
    return "\n".join(" ".join(line.split()) for line in "".join(scrambled_words).splitlines())

def substitute_sentence_percent(sentence, percent):
    words = re.split("(\W)", sentence)
    words_id = []
    for i in list(range(len(words))):
        letters = [c for c in words[i] if c.isalpha()]
        if len(letters) > 1:
            words_id.append(i)
    words_id_slc = random.sample(words_id, k=round(len(words_id)*percent))
    substituted_words = words
    for i in words_id_slc:
        substituted_words[i] = substitute_word(words[i])
    return "\n".join(" ".join(line.split()) for line in "".join(substituted_words).splitlines())

def scramble_sentence_keepfirst(sentence):
    words = re.split("(\W)", sentence)
    scrambled_words = [scramble_word_keepfirst(word) for word in words]
    return "\n".join(" ".join(line.split()) for line in "".join(scrambled_words).splitlines())

def scramble_sentence_keepfirstlast(sentence):
    words = re.split("(\W)", sentence)
    scrambled_words = [scramble_word_keepfirstlast(word) for word in words]
    return "\n".join(" ".join(line.split()) for line in "".join(scrambled_words).splitlines())

# scramble the sentence stepwise (e.g., 20% scrambled sentence = 10% scrambled sentence + another 10% scrambled words)
def scramble_sentence_percent_step(sentence, percent, scrambled_words_id, rate=0.1):
    step = round(percent / rate)
    words = re.split("(\W)", sentence)
    words_id = []
    for i in list(range(len(words))):
        letters = [c for c in words[i] if c.isalpha()]
        if len(letters) > 1:
            words_id.append(i)
    step_len = len(words_id)*rate
    words_id = sorted(list(set(words_id) - set(scrambled_words_id)))
    scrambled_words = words
    words_id_slc = random.sample(words_id, k=round(step_len*(step))-round(step_len*(step-1)))
    for i in words_id_slc:
        scrambled_words[i] = scramble_word(words[i])
    return "\n".join(" ".join(line.split()) for line in "".join(scrambled_words).splitlines()), (scrambled_words_id + words_id_slc)

def generate_demo_rec(samples, method, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    demos = []
    for i in samples:
        if method == "keepfirst":
            demo = scramble_sentence_keepfirst(i)
            demo = "Scrambled Sentence: " + demo + "\n" + "Recovered Sentence: " + scramble_sentence_percent(i, 0) + "\n\n"
            demos.append(demo)
        elif method == "keepfirstlast":
            demo = scramble_sentence_keepfirstlast(i)
            demo = "Scrambled Sentence: " + demo + "\n" + "Recovered Sentence: " + scramble_sentence_percent(i, 0) + "\n\n"
            demos.append(demo)
        else:
            demo = scramble_sentence_percent(i, method)
            demo = "Scrambled Sentence: " + demo + "\n" + "Recovered Sentence: " + scramble_sentence_percent(i, 0) + "\n\n"
            demos.append(demo)
    return "".join(demos)

def generate_demo_aqua(samples, percent, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    demos = []
    for i in samples:
        demo_q = scramble_sentence_percent(i["question"], percent)
        demo = "Question: " + demo_q + "\n" + "Choices: " + i["choice"] + "\nAnswer: " + i["answer"] + "\n\n"
        demos.append(demo)
    return "".join(demos)

