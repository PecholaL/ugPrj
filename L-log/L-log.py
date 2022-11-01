#! /Users/pecholalee/miniforge3/envs/py39 python3

''' 根据apache access日志训练模型用于对异常的检测
    程序流程如下:
        日志的初步筛选:提取出access日志中的URI
        数据清洗/向量化:找出URL参数等重要字段并进行向量化
        训练:使用多种模型进行训练用于与深度学习的方法对比
        测试:测试模型的准确度
'''

import time
from unittest import result
import numpy
import pandas
import urllib
import re
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import models
from keras import layers

# 数据集路径/名称
N_uri_file="normalURI.txt"
A_uri_file="anomalousURI.txt"
N_vector_file="normalVector.txt"
A_vector_file="anomalousVector.txt"

# SQL关键字
SQL_keyword=["and", "alter", "add", "by", "create", "delete", "drop", "from", "join", "insert", "into", "not", "or", "select", "set", "table", "union", "update", "where"]

# 提取URI
def uri_filter(rawLog, uriFile):
    print("Get timestamp and URI ...")
    # open输入输出文件
    Fin=open(rawLog, 'r')
    Fout=open(uriFile, 'w')
    lines=Fin.readlines()
    # 开始遍历日志每一行
    for line in lines:
        if "GET" in line:
            line=line.strip().split("\"")
            '''日期时间字符串
            # 格式：[27/Feb/2022:18:40:46 +0800]
            time_str=(line[0].split())[3]
            timestamp=time.mktime(time.strptime(time_str, "[%d/%b/%Y:%H:%M:%S"))
            '''
            # URI的提取与解析
            uri_raw=(line[1].split(" "))[1]
            uri=urllib.parse.unquote(uri_raw, encoding='ascii', errors='ignore')
            uri=uri.replace('\n', '').lower()
            Fout.writelines(uri+'\n')
    print(rawLog+" filtered!")
    Fin.close()
    Fout.close()


# 去除重复
def dlt_duplication(origFile, outputFile):
    print("Deleting duplication ...")
    Fin=open(origFile, 'r')
    origLines=Fin.readlines()
    with open(outputFile, 'a') as Fout:
        Fout.close()
    for line in origLines:
        Fout_r=open(outputFile, 'r')
        current=Fout_r.readlines()
        with open(outputFile, 'a') as Fout:
            if line not in current:
                Fout.write(line)
            Fout.close()
    Fin.close()
    print("Delete duplication completed!")


# 数据清洗(获取参数)/向量化
''' 提取特征, 向量化:
    0: URI长度
    1: 参数个数
    2: 参数总长度
    3: 参数最大长度
    4: 参数长度占URI长度比例
    5: 参数中数字的数量
    6: 参数中数字占比
    7: 参数中特殊符号的数量
    8: 参数中特殊符号占比
    9: 参数中空格的数量
    10: 参数中空格占比
    11: 参数中SQL关键词数量
    12: 服务端返回状态码
'''
def vectorize(uriFile, vectorFile):
    print("Data cleaning and vectorizing ...")
    Fin=open(uriFile, 'r')
    Fout=open(vectorFile, 'w')
    lines=Fin.readlines()
    vec=[]
    # 开始遍历每一条URI
    for line in lines:
        # 提取状态码
        #statusCode=line.split()[-1]
        # 提取URI
        #uri=line[:-len(statusCode)]
        ch0=len(line.strip()) # 0 URI长度
        # 解析参数
        line=urllib.parse.urlparse(line)
        params_dict=urllib.parse.parse_qs(line.query)
        # 统计参数个数
        paramCount=0
        # 将所有参数值合并在一个字符串中方便统计
        paramVal=""
        # 记录参数最大长度
        paramMaxLen=0
        for p in params_dict.values():
            paramCount+=1
            p=p[0]
            paramVal+=p
            if len(p)>paramMaxLen:
                paramMaxLen=len(p)
        ch1=paramCount # 1 参数个数
        ch2=len(paramVal) # 2 参数总长度
        ch3=paramMaxLen # 3 参数最大长度
        # 不含参数的情形
        if ch1==0:
            ch2=0
            ch3=0
            ch4=0
            ch5=0
            ch6=0
            ch7=0
            ch8=0
            ch9=0
            ch10=0
            ch11=0
        else:
            ch4=ch2/ch0 # 4 参数长度占URI长度比例
            ch5=len(re.findall("\d", paramVal)) # 5 参数中数字数量
            ch6=ch5/ch2 # 6 参数中数字占比
            ch9=len(re.findall(" ", paramVal)) # 9 参数中空格数量
            ch10=ch9/ch2 # 10 参数中空格占比
            # 统计参数中字母的数量
            charCount=len(re.findall(r"[a-z]", paramVal))
            ch7=ch2-ch5-ch9-charCount # 7 参数中特殊符号数量
            ch8=ch7/ch2 # 8 参数中特殊符号占比
            # 统计参数中SQL关键字出现数量
            kwCount=0
            for kw in SQL_keyword:
                kwCount+=paramVal.count(kw)
            ch11=kwCount # 11 参数中SQL关键字数量
        v=[ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11]
        Fout.write(str(v)+'\n')
        vec.append(v)
    print("Vectorizing cmopleted!")
    Fin.close()
    Fout.close()
    return vec


# 归一化
def normalization(vecArr):
    print("Normalization ...")
    X=numpy.array(vecArr)
    stdScl=StandardScaler()
    stdScl.fit(X)
    X_std=stdScl.transform(X)
    print("Normalization completed!")
    return X_std


# 训练及测试
# SVM 用于与深度学习对照
def SVM_train_test(X_train, X_test, y_train, y_test):
    # 选择模型
    cls=svm.LinearSVC(dual=False)
    # 训练模型
    print("Training(LinearSVC) ...")
    cls.fit(X_train, y_train)
    print("Training completed!")
    # 测试模型
    # 模型评估
    score=cls.score(X_test, y_test)
    y_pre=cls.predict(X_test)
    ret=classification_report(y_test, y_pre, labels=(0,1), target_names=["Normal", "Anomalous"])
    print("SVM test score:%.3f" % score)
    print(ret)


# 深度学习
def deepLearning(X_train, X_val, X_test, y_train, y_val, y_test):
    # 数据集尺寸
    print("size of train data: ", X_train.shape)
    # 采用序贯模型
    DLmodel=models.Sequential()
    # 四层之间全连接（Dense）
    # 1，2，3层激活函数为relu
    DLmodel.add(layers.Dense(12, activation='relu', input_shape=X_train.shape[1:]))
    for i in range(10):
        DLmodel.add(layers.Dense(32, activation='tanh'))
    # 4层激活函数为sigmoid
    DLmodel.add(layers.Dense(1, activation='sigmoid'))
    print("Compiling deep learning model ...")
    DLmodel.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
    print("Compile completed!")
    print("Training ...")
    # history保存了训练过程
    history=DLmodel.fit(X_train, y_train, epochs=50, batch_size=0, validation_data=(X_val, y_val))
    print("Training completed!")
    res=DLmodel.evaluate(X_test, y_test, batch_size=4096, verbose=0)
    print(res)
    y_pred=DLmodel.predict(X_test)
    con_matrix(y_pred, y_test)
    draw_ROC(y_test, y_pred)
    return history

# 制作混淆矩阵
def con_matrix(pre, y_test):
    res=[]  
    for i in range(len(pre)):
        if pre[i][0] > 0.5:
            p=1
        else:
            p=0
        res.append(p)
    print(str(pandas.crosstab(y_test, res, rownames=['labels'], colnames=['predicts'])))
    

# 绘制loss和accuracy曲线
def draw_loss(h):
    h_dict=h.history
    # LOSS
    loss_val=h_dict['loss']
    val_loss_val=h_dict['val_loss']
    epochs=range(1, len(loss_val)+1)
    plt.title('LOSS(Training and Validation)')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epochs, loss_val, label='train_loss', color='red')
    plt.plot(epochs, val_loss_val, label='val_loss', color='black')
    plt.legend()
    plt.show()
def draw_acc(h):
    h_dict=h.history
    # ACCURACY
    acc_val=h_dict['accuracy']
    val_acc_val=h_dict['val_accuracy']
    epochs=range(1, len(acc_val)+1)
    plt.title('ACCURACY(Training and Validation)')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(epochs, acc_val, label='train_accuracy', color='red')
    plt.plot(epochs, val_acc_val, label='val_accuracy', color='black')
    plt.legend()
    plt.show()

def draw_ROC(labels, pred):
    fpr1, tpr1, threshold1 = sklearn.metrics.roc_curve(labels, pred)  ###计算真正率和假正率
    roc_auc1 = sklearn.metrics.auc(fpr1, tpr1)  ###计算auc的值，auc就是曲线包围的面积，越大越好
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    # 筛选出原始日志中的URI
    #uri_filter(log_raw, uri_all)

    # 去除重复
    #dlt_duplication(N_uri_all, N_uri)
    #dlt_duplication(A_uri_all, A_uri)

    # 进行URI解析并进行向量化
    N_vector=vectorize(N_uri_file, N_vector_file)
    A_vector=vectorize(A_uri_file, A_vector_file)
    X=N_vector+A_vector

    # 归一化
    X_std=normalization(X)

    # 标签
    N_label=[0]*len(N_vector)
    A_label=[1]*len(A_vector)
    y=N_label+A_label

    # 训练及测试
    # SVM
#    print("--------------------SVM--------------------")
    # 划分训练集、测试集(6:4)
#    X_train, X_test, y_train, y_test=train_test_split(X_std, y, test_size=0.4, random_state=0)
#    SVM_train_test(X_train, X_test, y_train, y_test)


    # 深度学习
    print("----------------DeepLearning----------------")
    # 划分训练集、验证集、测试集(6:2:2)
    array_y=numpy.array(y)
    # 先划分出测试集
    X_t_v, X_test, y_t_v, y_test=train_test_split(X_std, array_y, test_size=0.2, random_state=0)
    # 再划分训练集和验证集
    X_train, X_val, y_train, y_val=train_test_split(X_t_v, y_t_v, test_size=0.25, random_state=0)
    H=deepLearning(X_train, X_val, X_test, y_train, y_val, y_test)
    # 绘制loss曲线判断是否过拟合
    draw_loss(H)
    draw_acc(H)



''' __END_OF_L-LOG__
    LI Pang-ching Apr.2022
'''