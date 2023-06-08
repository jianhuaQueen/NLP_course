import re
import collections
import math

import jieba

"""三元模型的信息熵"""
def get_trigram_entropy(bigram_tokens, trigram_tokens):

    bigram_frequency = dict(collections.Counter(bigram_tokens))
    trigram_frequency = collections.Counter(trigram_tokens)

    trigram_count = len(trigram_tokens)
    entropy = []
    for key, value in trigram_frequency.items():
        tri_fre = (1.0 * value) / trigram_count   # 计算联合概率P(x,y,z)
        bigram_fre = bigram_frequency[key[0:2]]
        log = math.log((1.0 * value / bigram_fre), 2)  # log函数的基底为2，单位为比特
        entropy.append(tri_fre * log)

    return round(sum(entropy) * (-1), 3)  # round，保留三位小数

def get_word_tokens():
    cn_stopwords = []  # 标点符号
    # 读取  cn_punctuation
    with open(file='cn_stopwords.txt', mode='r', encoding='utf-8') as file_obj:
        for line in file_obj:
            cn_stopwords.append(line.strip())

    tokens = []  # 词元
    words = []  # 字

    # 读取 语料库 文件列表
    with open(file='jyxstxtqj_downcc/inf.txt', mode='r', encoding='gbk') as file_obj:
        file_list = file_obj.readline().split(',')

    r1 = '(([a-zA-Z0-9]+)\.)*([a-zA-Z0-9]+)'  # 正则表达式
    for file_name in file_list:
        full_file_name = 'jyxstxtqj_downcc2/' + file_name + '.txt'
        with open(file=full_file_name, mode='r', encoding='gb18030') as file_obj:
            for line in file_obj:
                if line != '\n':
                    sentence = re.sub(r1, ' ', line.strip())  # 处理 www.cr173.com等英文字符
                    sentence = re.sub('(=+)', ' ', sentence)  # 处理=
                    # 获得字
                    for word in sentence:
                        if (word not in cn_stopwords) and word != ' ':
                            words.append(word)

                    # 获得词元
                    for token in jieba.cut(sentence=sentence):
                        if (token not in cn_stopwords) and token != ' ':  # 标点符号
                            tokens.append(token)


    return tokens, words


if __name__ == '__main__':
    tokens, words = get_word_tokens()

    # 以字为单位
    bigram_words = [pair for pair in zip(words[:-1], words[1:])]
    trigram_words = [pair for pair in zip(words[:-2], words[1:-1], words[2:])]
    word_entropy = get_trigram_entropy(bigram_words, trigram_words)  # 计算三元语法模型的熵

    # 以词为单位
    bigram_tokens = [pair for pair in zip(tokens[:-1], tokens[1:])]
    trigram_tokens = [pair for pair in zip(tokens[:-2], tokens[1:-1], tokens[2:])]
    token_entropy = get_trigram_entropy(bigram_tokens, trigram_tokens)  # 计算三元语法模型的熵

    print("语料库总字数:{}".format(len(words)))
    print("语料库总词数：{}".format(len(tokens)))
    print("三元语法模型字的信息熵:{}比特/字".format(word_entropy))
    print("三元语法模型词的信息熵:{}比特/词元".format(token_entropy))

