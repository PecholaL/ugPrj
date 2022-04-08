#! /Users/pecholalee/miniforge3/envs/py39 python3
'''
    对CSIC2010数据集进行学习
'''


import numpy, pandas
import urllib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm


# 原始数据集名称
test_file_normal_raw="normalTrafficTest.txt"
test_file_anomalous_raw="anomalousTrafficTest.txt"

# 预处理后的数据集名称
test_file_normal="test_normal.txt"
test_file_anomalous="test_anomalous.txt"


# 文本预处理
def preprocess(input_file, output_file):
    print("Preprocessing...")
    f=open(input_file, 'r', encoding='utf-8')
    lines=f.readlines()
    res=[]
    # 遍历每行
    for i in range(len(lines)):
        line=lines[i].strip()
        # 仅选择GET类型的数据的URL部分
        if line.startswith("GET"):
            res.append(line.split(" ")[1])
    f_output=open(output_file, 'w', encoding='utf-8')
    # 写入输出文件
    for line in res:
        # 解析URL
        line=urllib.parse.unquote(line, encoding='ascii', errors='ignore')
        line=line.replace('\n', '').lower()
        f_output.writelines(line+'\n')
    print("Preprocess of"+input_file+"Completed!")


# 读取原始数据集（已预处理）
def loadRawData(file):
    f=open(file, 'r', encoding='utf-8')
    lines=f.readlines()
    rawData=[]
    for line in lines:
        rawData.append(line)
    return rawData


# 向量化（TF-IDF）
def vectorize(rawData):
    v=TfidfVectorizer(min_df=0.0, analyzer="word", sublinear_tf=True)
    vecData=v.fit_transform(rawData)
    print("Vectorizing Completed!\n\tdata size:", vecData.shape)
    # 归一化
    stdScaler=sklearn.preprocessing.StandardScaler(with_mean=False)
    stdScaler.fit(vecData)
    vecData=stdScaler.transform(vecData)
    print("Normalization Completed!")
    return vecData


# SVM训练及测试
def SVM_train_test(train_X, train_y, test_X, test_y):
    # 选择模型
    cls=svm.LinearSVC(dual=False)
    # 训练模型
    print("Training...")
    cls.fit(train_X, train_y)
    print("Training Completed!")
    # 测试模型
    print("Test Score:%.2f" % cls.score(test_X, test_y))


if __name__=='__main__':
        # 预处理
        preprocess(test_file_normal_raw, test_file_normal)
        preprocess(test_file_anomalous_raw, test_file_anomalous)

        # 读取已预处理的原始数据
        normal_data_raw=loadRawData(test_file_normal)
        anomalous_data_raw=loadRawData(test_file_anomalous)

        # 标签
        label_normal=[0]*len(normal_data_raw)
        label_anomalous=[1]*len(anomalous_data_raw)
        L=label_normal+label_anomalous

        # 原始字符串数据向量化
        D=vectorize(normal_data_raw+anomalous_data_raw)

        # 划分测试集和训练集
        # D for data 、 L for label
        D_train, D_test, L_train, L_test=train_test_split(D, L, test_size=0.2, random_state=0)

        # 训练及测试
        SVM_train_test(D_train, L_train, D_test, L_test)


# 2022 LI Pang-ching