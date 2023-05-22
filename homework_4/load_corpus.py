import numpy as np
import jieba
import gensim
from sklearn.model_selection import train_test_split

new_words = ['韦蝠王', '阳教主', '吴国']
for word in new_words:
    jieba.add_word(word)

def load_novals(noval_path):
    """加载小说"""
    novals = []
    # 读取小说列表
    with open(file=noval_path + 'inf.txt', mode='r', encoding='gbk') as f:
        file_names = f.readline().split(',')
    f.close()

    # 读取小说
    # for index, name in enumerate(file_names):
    name = '鹿鼎记'
    for _ in range(1, 2):
        novel_name = noval_path + name + '.txt'
        sentences = list()
        with open(file=novel_name, mode='r', encoding='gb18030') as f:
            content = f.read()
            ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
            content = content.replace(ad, '')

            content = content.replace('（', '')
            content = content.replace('）', '')

            for paragraph in content.split('\n'):
                for sentence in paragraph.strip().split('。'):
                    if len(sentence) > 0:
                        sentences.append(jieba.lcut(sentence + '。'))

        novals.append(sentences)

    print("*************小说加载完毕*************")
    return novals


def pre_process_text(novels):
    # 获取source、target 文本pairs
    source_texts = list()
    target_texts = list()
    tokens = [['<eos>', '<bos>', '<pad>', '<unk>']]

    for novel in novels:
        source_texts.extend(novel[:-1])
        target_texts.extend(novel[1:])
        tokens.extend(novel)

    # 填充, eos, bos
    max_text_len = max(len(sentence) for novel in novels for sentence in novel) + 2
    source_padded = [['<bos>'] + sentence + ['<eos>'] + ['<pad>'] * (max_text_len - len(sentence)) for sentence in
                     source_texts]
    target_padded = [['<bos>'] + sentence + ['<eos>'] + ['<pad>'] * (max_text_len - len(sentence)) for sentence in
                     target_texts]

    # 构建词典
    vocab = gensim.corpora.Dictionary(tokens)
    # 将文本转化为向量
    source_encoded = [vocab.doc2idx(text) for text in source_padded]
    target_encoded = [vocab.doc2idx(text) for text in target_padded]

    print("*************语料数据预处理完成*************")
    return np.array(source_encoded), np.array(target_encoded), vocab, max_text_len

def get_dataset(noval_path):
    novels = load_novals(noval_path='jyxstxtqj_downcc/')
    source_encoded, target_encoded, vocab, max_text_len = pre_process_text(novels)
    X_train, X_test, y_train, y_test = train_test_split(source_encoded, target_encoded, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, vocab, max_text_len
