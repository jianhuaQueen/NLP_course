import random
import pandas as pd
import numpy as np
import jieba
import gensim
import matplotlib.pyplot as plt



# 读取中文停词
def load_cn_stopwords():
    cn_stopwords = []
    with open('cn_stopwords.txt', mode='r', encoding='utf-8') as f:

        cn_stopwords.extend([x.strip() for x in f.readlines()])
    f.close()
    cn_stopwords.extend(['道', '曰', '中', '便', '说', '说道', '～'])
    return cn_stopwords


def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
    content = content.replace(ad, '')
    return content


def load_novel(path):  # 读取语料内容
    para_list = []
    para_label = []

    # 读取小说列表
    with open(file=path + 'inf.txt', mode='r', encoding='gbk') as f:
        file_names = f.readline().split(',')
    f.close()
    # 读取小说
    for index, name in enumerate(file_names):
        novel_name = path + name +'.txt'
        with open(file=novel_name, mode='r', encoding='gb18030') as f:
            content = f.read()
            content = content_deal(content)
            for sentence in content.split('\n'):
                if len(sentence) < 500:
                    continue
                para_list.append(sentence)
                para_label.append(index)

    return para_list, para_label


if __name__ == '__main__':
    path = 'jyxstxtqj_downcc/'
    para_list, para_label = load_novel(path)

    # 加载停用词
    stop_words = load_cn_stopwords()

    # 均匀抽取200个段落
    text_ls = []
    text_label = []
    random_indices = random.sample(range(len(para_list)), 200)
    text_ls.extend([para_list[i] for i in random_indices])
    text_label.extend([para_label[i] for i in random_indices])

    # 分词，分别以字和词为基本单位
    tokens_word = []  # 以词文单位
    tokens_word_label = []
    tokens_char = []  # 以字为单位
    tokens_char_label = []
    for i, text in enumerate(text_ls):
        words = [word for word in jieba.lcut(sentence=text) if word not in stop_words]
        tokens_word.append(words)
        tokens_word_label.append(text_label[i])

        temp = []
        for word in words:
            temp.extend([char for char in word])
        tokens_char.append(temp)
        tokens_char_label.append(text_label[i])

    # 构造词典,为文档中的每个词分配一个独一无二的整数编号
    dictionary_word = gensim.corpora.Dictionary(tokens_word)
    dictionary_char = gensim.corpora.Dictionary(tokens_char)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus_word = [dictionary_word.doc2bow(tokens) for tokens in tokens_word]
    corpus_char = [dictionary_char.doc2bow(tokens) for tokens in tokens_char]

    # 训练lda模型，num_topics设置主题的个数
    results_perplexity_word = []
    results_cv_word = []
    results_perplexity_char = []
    results_cv_char = []
    num_topics_list = range(2, 10, 1)
    for num_topics in num_topics_list:
        # 以“词”作为基本单元
        lda_word = gensim.models.ldamodel.LdaModel(corpus=corpus_word, id2word=dictionary_word, num_topics=num_topics,
                                                   passes=20, alpha='auto', eta='auto')

        perplexity_word = -lda_word.log_perplexity(corpus_word)
        cv_model_word = gensim.models.CoherenceModel(model=lda_word, texts=tokens_word, dictionary=dictionary_word,
                                                     coherence='c_v')   # 一致性
        results_perplexity_word.append(perplexity_word)
        results_cv_word.append(cv_model_word.get_coherence())
        # 以“字”作为基本单元
        lda_char = gensim.models.ldamodel.LdaModel(corpus=corpus_char, id2word=dictionary_char, num_topics=num_topics,
                                                   passes=20, alpha='auto', eta='auto')
        perplexity_char = -lda_char.log_perplexity(corpus_char)
        cv_model_char = gensim.models.CoherenceModel(model=lda_char, texts=tokens_char, dictionary=dictionary_char,
                                                     coherence='c_v')   # 一致性
        results_perplexity_char.append(perplexity_char)
        results_cv_char.append(cv_model_char.get_coherence())

    # 创建画布
    fig, axes = plt.subplots(nrows=2, ncols=2)

    # 在第一个小区域中绘制第一条曲线
    axes[0, 0].plot(num_topics_list, results_perplexity_word, label='word')
    axes[0, 0].plot(num_topics_list, results_perplexity_char, label='char')
    axes[0, 0].set_title('perplexity')
    axes[0, 0].legend()

    # 在第二个小区域中绘制第二条曲线
    axes[0, 1].plot(num_topics_list, results_cv_word, label='word')
    axes[0, 1].plot(num_topics_list, results_cv_char, label='char')
    axes[0, 1].set_title('coherence')
    axes[0, 1].legend()

    # 在第三个小区域中绘制第三条曲线
    axes[1, 0].plot(num_topics_list, results_cv_word, label='coherence')
    axes[1, 0].plot(num_topics_list, results_perplexity_word, label='perplexity')
    axes[1, 0].set_title('word')
    axes[1, 0].legend()

    # 在第四个小区域中绘制第四条曲线
    axes[1, 1].plot(num_topics_list, results_cv_char, label='coherence')
    axes[1, 1].plot(num_topics_list, results_perplexity_char, label='perplexity')
    axes[1, 1].set_title('char')
    axes[1, 1].legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图像
    plt.show()
