
import os,jieba,io
from sklearn.naive_bayes import  MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# def tf_idf_t():
#     tfidf_vec = TfidfVectorizer()
#     documents = ['this is the bayes document','this is the second document','and the third one','is this the documnet']
#     tfidf_matric = tfidf_vec.fit_transform(documents)
#
#     #输出文档中所有不重复的词
#     print('不重复的词:',tfidf_vec.get_feature_names())
#     #输出每个单词对应的id
#     print('每个单词对应的id:',tfidf_vec.vocabulary_)
#     #输出每个单词在每个文档中的tf-idf值,向量里的顺序是按照词语的id顺序来的
#     tfidf_result = tfidf_matric.toarray()
#     print('每个单词tfidf值:',tfidf_result)
#
#
# '''
# 文档输入
# 开始阶段：对文档分词，加载停用词，计算单词的权重
# 分类阶段；生成分类器，分类器预测，计算正确率
# 注意：分词的数据准备：包括分词，单词权重的计算，去掉停用词
# '''
# import nltk   #对英文文档的分词器
# import jieba  #对中文文档的分词器
# def bayes_process():
#     text = ''
#
#     '''分词'''
#
#     #对英文进行分词
#     word_list = nltk.word_tokenize(text) #分档分词
#     nltk.pos_tag(word_list) #标注单词的词性
#
#     #对中文进行分词
#     word_list = jieba.cut(text)
#
#     '''加载停用词
#     需要自己读取停用的词表文件
#     '''
#     import io
#     stop_word_path = ''
#     stop_words = [line.strip().decode('utf-8') for line in io.open(stop_word_path).readlines()]
#
#     '''计算单词的权重
#         得到tf-idf的特征空间features
#         max_df 代表一个单词在50%的文章中都出现过，因此就不作为分词统计了
#         一般很少设置min_df,因为min_df通常很小
#     '''
#     tf= TfidfVectorizer(stop_words=stop_words,max_df=0.5)
#     featrues = tf.fit_transform(train_contents)
#
#
#     '''生成朴素贝叶斯分类器
#         1) 当alpha=1,时，使用的是laplace（加法平滑）平滑，laplace平滑采用的是加1的方式，来统计没有出现过的单词的频率，这样当训练样本很大的时候
#             ，加1得到的概率变化可以忽略不计，也同时避免了零概率的问题
#             对事件假设做了一个统一的先验概率（也就是每个n元组都有相同的可能性），事实上就是贝叶斯估计；laplace法则给出的估计方法依赖于词表的大小
#             对于一个大词表上的稀疏数据集，laplace法则实际上把太多的概率转移到了未知事件上。
#         2）当0<alpha<1,使用的是Lidstone平滑，对于lidstone平滑来说，alpha越小，迭代的次数越多，精度越高。
#             统计实践中通常的解决多项式估计问题的方法是连续的Lidstone法则，而不是加1，而是加一个正值，通常较小。这种也称为期望似然估计，
#             通过一个小的正值，避开缺点：（1）太多的概率空间被转移到未知的事件上；但是还是有自身的缺点：1）需要预先猜测一个合适的正值，
#             2）使用lidstone法则的折扣总是在最大似然估计频率上给出一个线性的概率估计。但是这和低频情况下的经验分布不能很好的吻合。
#
#     '''
#     from sklearn.naive_bayes import  MultinomialNB
#     clf = MultinomialNB(alpha=0.001).fit(train_features,train_labels)
#
#
#     '''
#     使用生成的分类器做预测
#     先用训练集的分词创建一个分类器，在对测试集的数据进行拟合，得到测试集的特征矩阵
#     然后用训练好的分类器对新数据做预测
#     '''
#     test_tf = TfidfVectorizer(stop_words=stop_words,max_df = 0.5,vocabulary=train_vocabulary)
#     test_features = test_tf.fit_transform(test_content)
#     predicted_labels = clf.predict(test_features)
#
#     '''计算准确率
#         1）使用metric中的accuracy_score,实现对实际的结果和预测的结果做对比，给出模型的准确率
#     '''
#     from sklearn import metrics
#     print(metrics.accuracy_score(test_labels,predicted_labels))

def load_stop_word(stop_path):
    stop_words = [line.strip() for line in  io.open(stop_path,mode='r',encoding='utf-8').readlines()]
    return stop_words

def load_data(base_path):
    '''
    :param base_path: 基础路径
    :return: 分词的列表，标签列表
    '''
    documnets = []
    labels = []
    for root,dirs,files in os.walk(base_path):

        for dir in dirs:
            dir_path = root+'/'+dir+'/'
            files = os.listdir(dir_path)
            for file in files:
                labels.append(dir)
                file_path = dir_path+file
                with open(file_path,'rb') as rh:
                    content = rh.read()
                    wordlist = list(jieba.cut(content))
                    documnets.append(' '.join(wordlist))
    return documnets,labels

def bayes_T():
    #加载停用的词
    stop_path = r'F:\blog\blogApp\data_analysis\data\text_classification-master\text classification\stop\stopword.txt'
    stop_word_list = load_stop_word(stop_path)

    #分词
    train_base_path = r'F:\blog\blogApp\data_analysis\data\text_classification-master\text classification\train'
    train_data,train_labels = load_data(train_base_path)

    test_base_path = r'F:\blog\blogApp\data_analysis\data\text_classification-master\text classification\test'
    test_data,test_labels = load_data(test_base_path)

    #计算矩阵
    tt = TfidfVectorizer(stop_words=stop_word_list,max_df=0.5)
    tf= tt.fit_transform(train_data)

    #训练模型
    clf = MultinomialNB(alpha=0.001).fit(tf,train_labels)

    #模型预测
    test_tf = TfidfVectorizer(stop_words=stop_word_list,max_df =0.5,vocabulary=tt.vocabulary_)
    test_features = test_tf.fit_transform(test_data)
    predicted_labels = clf.predict(test_features)

    #获得结果
    print(accuracy_score(test_labels,predicted_labels))








if __name__=='__main__':
    bayes_T()
