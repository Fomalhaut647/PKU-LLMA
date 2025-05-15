from bpe import Tokenizer
from transformers import GPT2Tokenizer

my_tokenizer = Tokenizer()
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text_1 = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."

text_2 = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"


def train_tokenizer():
    with open("./hw1-code/bpe/manual.txt", "r") as f:
        text = f.read()
    my_tokenizer.train(text, 1024)


def checkout_bpe():
    with open("./hw1-code/bpe/manual.txt", "r") as f1, open("./hw1-code/bpe/manual_encoded_decoded.txt", "w") as f2:
        text = f1.read()
        f2.write(my_tokenizer.decode(my_tokenizer.encode(text)))

    with open("./hw1-code/bpe/manual.txt", "r") as f1, open("./hw1-code/bpe/manual_encoded_decoded.txt", "r") as f2:
        original = f1.read()
        encoded = f2.read()
        print("Files are identical:", original == encoded)


def my_encode():
    tokens_1 = my_tokenizer.encode(text_1)
    tokens_2 = my_tokenizer.encode(text_2)
    return tokens_1, tokens_2


def hf_encode():
    tokens_1 = hf_tokenizer.encode(text_1)
    tokens_2 = hf_tokenizer.encode(text_2)
    return tokens_1, tokens_2


if __name__ == "__main__":
    train_tokenizer()

    checkout_bpe()
    
    print("--------------------------------")
    my_tokens_1, my_tokens_2 = my_encode()
    hf_tokens_1, hf_tokens_2 = hf_encode()
    print("my_tokens_length:", len(my_tokens_1), len(my_tokens_2))
    print("hf_tokens_length:", len(hf_tokens_1), len(hf_tokens_2))

