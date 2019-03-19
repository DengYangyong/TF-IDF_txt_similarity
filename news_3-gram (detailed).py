import re,os,gc 
from jpype import *  
import numpy as np

from itertools import chain  

from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer  

import matplotlib.pyplot as plt   
from wordcloud import WordCloud  

from scipy.spatial.distance import pdist   

#正则表达式用于去掉特殊符号
pattern=re.compile(u'[0-9a-zA-Z\u4E00-\u9FA5]')
pattern1 = re.compile(r'[0-9]')

# 调用hanlp之前的准备工作
root_path="/home/dyy/Action_in_nlp"
djclass_path="-Djava.class.path="+root_path+os.sep+"hanlp"+os.sep+"hanlp-1.7.1.jar:"+root_path+os.sep+"hanlp"
startJVM(getDefaultJVMPath(),djclass_path,"-Xms1g","-Xmx1g")
Tokenizer = JClass('com.hankcs.hanlp.tokenizer.StandardTokenizer')

def tokenizer_hnlp(sentence):
     
    drop_pos_set=set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh'])
    segs = [(word_pos_item.toString().split('/')[0],word_pos_item.toString().split('/')[1]) for word_pos_item in Tokenizer.segment(sentence)]   
    seg_filter = [word_pos_pair[0] for word_pos_pair in segs if len(word_pos_pair)==2 and word_pos_pair[0]!=' ' and word_pos_pair[1] not in drop_pos_set]
    return seg_filter
    # drop_pos_set: 是一些停用词的词性列表，这是通过词性来过滤停用词的一种思路，非常有意思。词性的意思可看计算所词性标注集。
    # segs ： 这段非常长，其实word_pos_item 就是类似 '证监会/n' 这样的一个词与词性的pair，不过不是字符串，所以要toString()转为字符串后，用'/'来拆开。 
    # segs的输出：[('解局', 'nr'), ('易会满', 'nr'), ('首', 'q'), ('秀', 'ag'), ('今天', 't'), ('27', 'm'), ('日', 'b'),...]
    # seg_filter: 这段更长，耐心点兄弟，就是segs的每个元素如果长度大于2，且词性不是停用词，就把词取出来。
    # seg_filter的输出为：['解局', '易会满', '首', '秀', '今天', '27', '日', '下午', '新任', '证监会', '主席', '易会满',...]

def list_3_ngram(sentence, n=3, m=2):
    if len(sentence) < n:
        n = len(sentence)
    temp=[sentence[i - k:i] for k in range(m, n + 1) for i in range(k, len(sentence) + 1) ]
    return [item for item in temp if len(''.join(item).strip())>0 and len(pattern1.findall(''.join(item).strip()))==0]
    # temp: [['解局', '易会满'], ['易会满', '首'], ['首', '秀'], ['秀', '今天'], ['今天', '27'], ['27', '日'], ['日', '下午']...['最后', '是', '敬畏'], ['是', '敬畏', '风险']] 
    #这里用到pattern1这个正则表达式去掉数字，包含了数字的3-gram，就过滤掉,['今天', '27'], ['27', '日'] 就被整个过滤掉了。

def pictures(text_list):
    for i,text in enumerate(text_list):
        text = " ".join(text)        
        wc = WordCloud(
            font_path='/home/dyy/Downloads/font163/simhei.ttf',     #字体路劲
            background_color='white',   #背景颜色
            width=1000,
            height=600,
            max_font_size=50,            #字体大小
            min_font_size=10,
            mask=plt.imread('picture/cow.jpg'),  #背景图片
            max_words=1000
        )
        wc.generate(text)
        wc.to_file(str(i)+'.png')    #图片保存

        #显示图片
        plt.figure('The'+str(i)+'picture')   #图片显示的名字
        plt.imshow(wc)
        plt.axis('off')        #关闭坐标
        plt.show()

if __name__=="__main__":   
    
    c_root = os.getcwd()+os.sep+"cnews"+os.sep  # 文档所在的路径
    news_list = []
    for file in os.listdir(c_root): 
            fp = open(c_root+file,'r',encoding="utf8")
            news_list.append(list(fp.readlines()))  # 每篇文档放在一个列表中, 读取的时候顺序倒了，第一列是上面网页中的第三篇，第三列是第一个网页。
            # 第二列的内容news_list[1] ：['证券日报\n', '02-2808:25\n', '\n', '易会满首秀“施政理念” 聚焦“市场化”\n', '\n', '■董文\n', '\n', '昨天，证监会主席易会...]
     
    # news_list[1] : '证券日报02280825易会满首秀施政理念聚焦市场化董文昨天证监会主席易会满...'
    # 看起来有点复杂哈，慢慢看，用了两个列表循环，对于每篇文档中的每行文本，用正则表达式去除非字母数字和中文的内容
    # 然后把每行的内容拼接起来，用了两次join。
    # news_list 的元素是三篇文档，['解局易会满...','证券日报...','信息量巨大...']。
    news_list = [''.join([''.join(pattern.findall(sentence)) for sentence in text ]) for text in news_list]
    
    copus=[tokenizer_hnlp(line.strip()) for line in news_list]
    
    doc=[]
    if len(copus)>1: 
        for list_copus in copus:
            doc.extend([' '.join(['_'.join(i) for i in list_3_ngram(list_copus,n=3, m=2)])])
            # doc[0] 的结果为： '解局_易会满 易会满_首 首_秀 秀_今天 日_下午 下午_新任 新任_证监会...'
            # n如果取4就是4元语法模型，m=1时可以把单个词也取到。
    
    vectorizer =CountVectorizer()              # 该类用来统计每篇文档中每个词语的词频                        
    transformer=TfidfTransformer()           #该类会统计每篇文档中每个词语的tf-idf权值  
    freq=vectorizer.fit_transform(doc)       #freq.toarray() 可以得到词频矩阵，是3*8519的矩阵，因为3篇文档的3-gram一共得到了8519个单元。
    tfidf=transformer.fit_transform(freq)   # tfidf.toarray() 可以得到tfidf 矩阵，是3* 8519的矩阵
     
    #tfidf_dic=vectorizer.get_feature_names()
    tfidf_dic=vectorizer.vocabulary_         
    # 得到的是{'解局_易会满': 7257, '易会满_首': 5137, '首_秀': 8451, '秀_今天': 6516, ...}
    # 字典中的value表示的是这个词汇在语料库词汇表中对应的索引。
    
    tfidf_dic=dict(zip(tfidf_dic.values(),tfidf_dic.keys()))
    # 得到：{7257: '解局_易会满', 5137: '易会满_首', 8451: '首_秀', 6516: '秀_今天',..},对上面的字典进行键值反转，方便下面根据索引取到词汇，取关键词。
  
    index_keyword =[]
    for i, tfidf_i in enumerate(tfidf.toarray()):  
        index_keyword.append([(j,value) for j, value in enumerate(tfidf_i)])
    # 这里有点复杂，先看输出：index_keyword[0] ：[(118, 0.03470566583648118), (324, 0.03470566583648118), (576, 0.052058498754721766),..]
    # 上面是第一篇文本的每个词对应的索引及其TF-IDF值，索引和TF-IDF做成了一个元组，方便后面根据TF-IDF进行排序。
    index_keyword = [sorted(i,key=lambda x:x[1],reverse=True) for i in index_keyword]
    #index_keyword得到： [[(7557, 0.2459727038614817), (6533, 0.16398180257432113), (6002, 0.10248862660895071),...],[...],[...]] ,
    # 每篇文档的字的索引按TF-IDF值进行降序排列。
    
    index_keyword = [[j[0] for j in i] for i in index_keyword]
    #把索引单独拎出来，[[7557, 6533, 6002, 3005, 5061, 576, 2627, 3524,...],[...],[...]]
    # 也就是说7557对应的词是第一篇文档中TF-IDF值最高的！
    
    list_keyword= []
    for i in index_keyword:
        list_keyword.append([tfidf_dic[j] for j in i])  # 还记得上面那个反转的字典么，这里用来根据索引取关键词。
    list_keyword = [i[:30] for i in list_keyword]  
    # 这里对关键词列取前30，就能得到TOP30的关键词了。
    # list_keyword ：[['资本_市场', '科创_板', '注册_制', '好_企业', ...],[['资本_市场', '科创_板', '注册_制', '易会满_表示'...],[['资本_市场', '科创_板', '上_表示', '新闻发布会_上_表示'...]
    
    pictures(list_keyword)
    #去重复并保持顺序。
    set_keyword = list(chain.from_iterable(list_keyword))  # 把3个列表中的各30个关键词合成一个列表，一共90个关键词，有重复的。
    set_keyword = sorted(set(set_keyword),key=set_keyword.index)  #用集合这种格式来去除重复，然后保持原来的顺序，不打乱，实际得到74个关键词。
    #得到 ['资本_市场', '科创_板', '注册_制', '好_企业', '易会满_今天', '个_敬畏', '四_个_敬畏', '市场_上', '市场_违法', '市场_违法_违规',...]
    
    
    freq_keyword = np.zeros(shape=(3,len(set_keyword)))  # 构造一个元素为0的词频矩阵。
    for i,txt in enumerate(doc):
        for word in txt.split():   # i为0时，txt为'解局_易会满 易会满_首 首_秀 秀_今天 日_下午 下午_新任 新任_证监会...'
            if word in set_keyword:  # 第一个word为 '解局_易会满'
                freq_keyword[i,set_keyword.index(word)] += 1  # 如果word在词汇表中，那么相应的位置就加1。
    # freq_keyword[0] 为：array([24., 16., 10.,  5.,  4.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,...])
                
    cos_12 = 1-pdist(np.vstack([freq_keyword[0],freq_keyword[1]]),'cosine')   # 计算第1和第2篇文章的余弦距离
    cos_13 = 1-pdist(np.vstack([freq_keyword[0],freq_keyword[2]]),'cosine')   # 计算第1和第3篇文章的余弦距离
    cos_23 = 1-pdist(np.vstack([freq_keyword[1],freq_keyword[2]]),'cosine')   # 计算第2和第3篇文章的余弦距离
    print([cos_12,cos_13,cos_23])
    # 得到余弦距离：[array([0.83047182]), array([0.7130633]), array([0.81106869])]
    
    fp.close()
    shutdownJVM()
                      

        
                
    