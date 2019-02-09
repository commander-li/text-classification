import os, sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# ͣ�ôʱ��ַ
stop_word_path = 'F:/Python/notebook/text classification/stop/stopword.txt'
# ѵ������Ŀ¼
train_base_path = 'F:/Python/notebook/text classification/train/'
# ���Լ���Ŀ¼
test_base_path = 'F:/Python/notebook/text classification/test/'

# �ĵ����
train_labels = ['����', 'Ů��', '��ѧ', 'У԰']
test_labels = ['����', 'Ů��', '��ѧ', 'У԰']

def get_data(base_path, labels):   #��ȡ����
    contents = []
    # ���������ĵ���ȡ�ļ�����
    for label in labels:
        files = {fileName for fileName in os.listdir(base_path + label)}
        try:
            for fileName in files:
                file = open(base_path + label + '/' + fileName, encoding='gb18030') 
                word = jieba.cut(file.read())  #ʹ��jieba���������ĵ����зִ�
                contents.append(' '.join(word))  # �дʣ��ÿո�ָ�
        except Exception:
            print(fileName + '�ļ���ȡ����')
    return contents

# 1�����ĵ����зִ�
# ��ȡѵ����
train_contents = get_data(train_base_path, train_labels)
# print(train_contents)
# print('ѵ�����ĳ��ȣ�', len(train_contents))

# ��ȡ���Լ�
test_contents = get_data(test_base_path, test_labels)
# print(test_contents)
# print('ѵ�����ĳ��ȣ�', len(test_contents))

# 2������ͣ�ôʱ�
stop_words = [line.strip() for line in open(stop_word_path, encoding = 'utf-8-sig').readlines()]
# print(stop_words)

# 3�����㵥��Ȩ��
tf  = TfidfVectorizer(stop_words = stop_words, max_df = 0.5)
train_features = tf.fit_transform(train_contents)

# 4���������ر�Ҷ˹��������ʹ�ö���ʽ��Ҷ˹������
train_labels = ['����'] * 1337 + ['Ů��']*954 + ['��ѧ'] * 766 + ['У԰']*249
clf = MultinomialNB(alpha = 0.001).fit(train_features, train_labels)  # MultinomialNB.fit(x,y) ���� y������Ҫ����ѵ��x������

# 5��ʹ�����ɵķ�������Ԥ��
test_tf = TfidfVectorizer(stop_words = stop_words, max_df=0.5, vocabulary=tf.vocabulary_)
test_features = test_tf.fit_transform(test_contents)

# Ԥ��
predicted_labels = clf.predict(test_features)

# 6������׼ȷ��
test_labels = ['����']*115 + ['Ů��']*38 + ['��ѧ']*31 + ['У԰']*16
print('׼ȷ��', metrics.accuracy_score(test_labels, predicted_labels))

׼ȷ�� 0.91