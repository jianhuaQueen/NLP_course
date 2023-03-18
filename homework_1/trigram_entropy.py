import re
import collections
import math

import jieba

sentence_boundary_token = "<space>"
unknown_token = "<unk>"

"""三元模型的信息熵"""
def get_trigram_entropy(bigram_tokens, trigram_tokens):

    bigram_frequency = dict(collections.Counter(bigram_tokens))
    trigram_frequency = collections.Counter(trigram_tokens)

    trigram_count = len(trigram_tokens)
    bigram_count = len(bigram_tokens)
    entropy = []
    for key, value in trigram_frequency.items():
        tri_fre = (1.0 * value) / trigram_count
        log = math.log((tri_fre * bigram_count / bigram_frequency[key[0:2]]), 2)  # log函数的基底为2，单位为比特
        entropy.append(tri_fre * log)
    return round(sum(entropy) * (-1), 3)  # round，保留三位小数

def get_tokens():
    cn_punctuation = []  # 标点符号
    # 读取  cn_punctuation
    with open(file='cn_punctuation.txt', mode='r', encoding='utf-8') as file_obj:
        for line in file_obj:
            cn_punctuation.append(line.strip())

    tokens = []  # 词元
    word_count = 0  # 字数

    # 读取 语料库 文件列表
    with open(file='jyxstxtqj_downcc/inf.txt', mode='r', encoding='gbk') as file_obj:
        file_list = file_obj.readline().split(',')

    r1 = '(([a-zA-Z0-9]+)\.)*([a-zA-Z0-9]+)'  # 正则表达式
    for file_name in file_list:
        full_file_name = 'jyxstxtqj_downcc/' + file_name + '.txt'
        with open(file=full_file_name, mode='r', encoding='gb18030') as file_obj:
            for line in file_obj:
                if line != '\n':
                    sentence = re.sub(r1, ' ', line.strip())  # 处理 www.cr173.com等英文字符
                    for token in jieba.cut(sentence=sentence):
                        if token in cn_punctuation:  # 标点符号
                            if(len(tokens) ==0 or tokens[-1] != sentence_boundary_token):
                                tokens.append(sentence_boundary_token)
                        elif token == ' ':    # 非中文字符
                            if (len(tokens) ==0 or tokens[-1] != unknown_token):
                                tokens.append(unknown_token)
                        else:
                            tokens.append(token)
                            word_count += len(token)

    return tokens, word_count


if __name__ == '__main__':
    tokens, word_count = get_tokens()

    bigram_tokens = [pair for pair in zip(tokens[:-1], tokens[1:])]
    trigram_tokens = [pair for pair in zip(tokens[:-2], tokens[1:-1], tokens[2:])]

    entropy = get_trigram_entropy(bigram_tokens, trigram_tokens)  # 计算三元语法模型的熵
    entropy = round(entropy/3, 3)

    print("语料库总字元数:{}".format(word_count))
    print("词元token数量：{}".format(len(tokens)))
    print("三元词元组的数量：{}".format((len(trigram_tokens))))
    print("三元语法模型的信息熵:{}比特/词元".format(entropy))


