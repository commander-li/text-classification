import os, sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# 停用词表地址
stop_word_path = 'F:/Python/notebook/text classification/stop/stopword.txt'
# 训练集根目录
train_base_path = 'F:/Python/notebook/text classification/train/'
# 测试集根目录
test_base_path = 'F:/Python/notebook/text classification/test/'

# 文档类别
train_labels = ['体育', '女性', '文学', '校园']
test_labels = ['体育', '女性', '文学', '校园']

def get_data(base_path, labels):   #获取数据
    contents = []
    # 遍历各个文档获取文件内容
    for label in labels:
        files = {fileName for fileName in os.listdir(base_path + label)}
        try:
            for fileName in files:
                file = open(base_path + label + '/' + fileName, encoding='gb18030') 
                word = jieba.cut(file.read())  #使用jieba包对中文文档进行分词
                contents.append(' '.join(word))  # 切词，用空格分隔
        except Exception:
            print(fileName + '文件读取错误')
    return contents

# 1、对文档进行分词
# 获取训练集
train_contents = get_data(train_base_path, train_labels)
# print(train_contents)
# print('训练集的长度：', len(train_contents))

# 获取测试集
test_contents = get_data(test_base_path, test_labels)
# print(test_contents)
# print('训练集的长度：', len(test_contents))

# 2、加载停用词表
stop_words = [line.strip() for line in open(stop_word_path, encoding = 'utf-8-sig').readlines()]
# print(stop_words)

# 3、计算单词权重
tf  = TfidfVectorizer(stop_words = stop_words, max_df = 0.5)
train_features = tf.fit_transform(train_contents)

# 4、生成朴素贝叶斯分类器，使用多项式贝叶斯分类器
train_labels = ['体育'] * 1337 + ['女性']*954 + ['文学'] * 766 + ['校园']*249
clf = MultinomialNB(alpha = 0.001).fit(train_features, train_labels)  # MultinomialNB.fit(x,y) 其中 y的数量要等于训练x的数量

# 5、使用生成的分类器做预测
test_tf = TfidfVectorizer(stop_words = stop_words, max_df=0.5, vocabulary=tf.vocabulary_)
test_features = test_tf.fit_transform(test_contents)

# 预测
predicted_labels = clf.predict(test_features)

# 6、计算准确率
test_labels = ['体育']*115 + ['女性']*38 + ['文学']*31 + ['校园']*16
print('准确率', metrics.accuracy_score(test_labels, predicted_labels))

## 准确率 0.91
