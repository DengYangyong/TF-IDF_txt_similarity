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

# 用hanlp进行分词和根据词性去掉停用词
def tokenizer_hanlp(sentence):
    
    drop_pos_set=set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh'])
    segs = [(word_pos_item.toString().split('/')[0],word_pos_item.toString().split('/')[1]) for word_pos_item in Tokenizer.segment(sentence)]   
    seg_filter = [word_pos_pair[0] for word_pos_pair in segs if len(word_pos_pair)==2 and word_pos_pair[0]!=' ' and word_pos_pair[1] not in drop_pos_set]
    return seg_filter

# 生成3-gram
def list_3_ngram(sentence, n=3, m=2):
    if len(sentence) < n:
        n = len(sentence)
    temp=[sentence[i - k:i] for k in range(m, n + 1) for i in range(k, len(sentence) + 1) ]
    return [item for item in temp if len(''.join(item).strip())>0 and len(pattern1.findall(''.join(item).strip()))==0]

# 绘制词云图    
def pictures(text_list):
    for i,text in enumerate(text_list):
        text = " ".join(text)        
        wc = WordCloud(font_path='/home/dyy/Downloads/font163/simhei.ttf',     
            background_color='white',   
            width=1000, height=600,
            max_font_size=50,            
            min_font_size=10,
            mask=plt.imread('picture/cow.jpg'),  
            max_words=1000 )
        wc.generate(text)
        wc.to_file(str(i)+'.png')    
        
        plt.figure('The'+str(i)+'picture')  
        plt.imshow(wc)
        plt.axis('off')        
        plt.show()

if __name__=="__main__":   
    #读取文本并进行分词
    c_root = os.getcwd()+os.sep+"cnews"+os.sep #
    news_list = []
    for file in os.listdir(c_root):
            fp = open(c_root+file,'r',encoding="utf8")
            news_list.append(list(fp.readlines()))
            
    news_list = [''.join([''.join(pattern.findall(sentence)) for sentence in text ]) for text in news_list]
    copus=[tokenizer_hanlp(line.strip()) for line in news_list]
    
    # 生成3-gram
    doc=[]
    if len(copus)>1: 
        for list_copus in copus:
            doc.extend([' '.join(['_'.join(i) for i in list_3_ngram(list_copus,n=3, m=2)])])
     
    # 计算词频和IF-IDF ，得到词汇表及其索引       
    vectorizer =CountVectorizer()                          
    transformer=TfidfTransformer()
    freq=vectorizer.fit_transform(doc)  
    tfidf=transformer.fit_transform(freq)
        
    #tfidf_dic=vectorizer.get_feature_names()
    tfidf_dic=vectorizer.vocabulary_   
    tfidf_dic=dict(zip(tfidf_dic.values(),tfidf_dic.keys()))
    
    #得到每篇文档TOP30的关键词。
    index_keyword =[]
    for i, tfidf_i in enumerate(tfidf.toarray()):  
        index_keyword.append([(j,value) for j, value in enumerate(tfidf_i)])
    index_keyword = [sorted(i,key=lambda x:x[1],reverse=True) for i in index_keyword]

    index_keyword = [[j[0] for j in i] for i in index_keyword]
    
    list_keyword= []
    for i in index_keyword:
        list_keyword.append([tfidf_dic[j] for j in i])
    list_keyword = [i[:30] for i in list_keyword]  
    
    # 画词云图
    pictures(list_keyword)
    
    # 合并得到关键词词汇表
    set_keyword = list(chain.from_iterable(list_keyword))
    set_keyword = sorted(set(set_keyword),key=set_keyword.index)
    
    # 统计各篇文档相对于关键词词汇表的词频矩阵
    freq_keyword = np.zeros(shape=(3,len(set_keyword)))
    for i,txt in enumerate(doc):
        for word in txt.split():
            if word in set_keyword:
                freq_keyword[i,set_keyword.index(word)] += 1
                
     #计算文档之间的余弦距离           
    cos_12 = 1-pdist(np.vstack([freq_keyword[0],freq_keyword[1]]),'cosine')
    cos_13 = 1-pdist(np.vstack([freq_keyword[0],freq_keyword[2]]),'cosine') 
    cos_23 = 1-pdist(np.vstack([freq_keyword[1],freq_keyword[2]]),'cosine')
    print([cos_12,cos_13,cos_23])
    fp.close()
    shutdownJVM()
                      

        
                
    