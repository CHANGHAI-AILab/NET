import pandas as pd
import numpy as np
import os


class reader:
    def FileReader(self):
        pass
class csv_Reader(reader):

    data_test={}
    def __init__(self,path,col1,col2):

        self.path=path

        #起始列和终止列
        self.col1=col1
        self.col2 =col2
        #最终行数


    def count(self):
        data1 = pd.read_csv(self.path, on_bad_lines="skip")
        data1=data1.fillna(0)
        list1 = []
        for i in range(self.col1, self.col2):
            list1.append(i)

        rows = len(data1)
        data3 = data1.iloc[0:rows, list1]

        data_all={}
        data_all=pd.DataFrame(data_all)

        for i in data3:


            data_max=np.max(data3[i])
            data_min=np.min(data3[i])
            data_mean=np.mean(data3[i])
            list1=[1,data_min,data_max,data_mean]
            list2=["_min","_max","_mean"]
            #后缀t
            for t in range(1,4):
                data_num = list1[t]
                str_end=list2[t-1]


                data_col=i+str_end
                data_cols={data_col:[f"{data_num}"]}
                data_cols=pd.DataFrame(data_cols)
                data_all=pd.concat([data_all,data_cols],axis=1)

        return data_all

class get_word:
    def __init__(self,path):
        self.path=path


#获得文件名字
    def get_all_files(self):
        file_names = []

        for file_name in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, file_name)):

                   file_names.append(file_name)
        return file_names


#获得文件夹名字

    def get_folders(self):
        folders = []
        for item in os.listdir(self.path):
            item_path = os.path.join(self.path, item)
            if os.path.isdir(item_path):
                folders.append(item)
        return folders



#对接受所有数据的项进行容器处理，就将其转换为可以存储为csv的模式
data_all={}
data_all=pd.DataFrame(data_all)




if __name__ == '__main__':
    # data2=pd.read_csv("1940017C2_Feats_I.csv")
    # print(data2)

    g = 0


    #将文件名分为三部分进行连接
    list1=["_Feats_I.csv",'_Feats_S.csv','_Feats_T.csv']
    data_csv_name = get_word(r'D:\feature')

    #name获取feature里所有的id
    name = data_csv_name.get_folders()
    len_name=len(name)



    # 这里的len_name指的是list长度---->list中的每一个数据代表一个文件夹名字
    for i in range(0,len_name):

        #测试运行的长度，本系统的长度为1503，防止中间报错或者进行时间的判定
        g+=1
        # print(g)



        #利用数组查询的方式进行各个数据名称的遍历
        str1=name[i].split('_')
        ID=str1[0]
        # print(ID)




        #每一个数据都需要一个表头，即id与文件名组成的数据
        # 加入ID
        data_index={'ID':ID}




        #这里进行横向连接，将数据加入每一个数据的表头中，axis=0
        data_index=pd.DataFrame(data_index,index=[0])




        #判断下列数据是否有四个文件
        if_path1=r'D:\feature'
        if_path2=rf'\{ID}'
        if_path=if_path1+if_path2

        data_csv_name_if = get_word(if_path)
        name_if = data_csv_name_if.get_all_files()
        if len(name_if)!=4:
            #表示4的直接跳过，不进行以下循环
            continue

        data_all_for = {}
        data_all_for = pd.DataFrame(data_all_for)
        data_all_for=pd.concat([data_index,data_all_for],axis=1)


        for j in range(0,3):
            #路径组合
            a=r"D:\feature"
            b=rf"\{name[i]}" + rf"\{name[i]}{list1[j]}"
            name_csv=a+b

            j=str(j)
            data=csv_Reader(name_csv, 4, 64)
            data_axis=data.count()
            data_all_for=pd.concat([data_all_for,data_axis],axis=1)
        data_all=pd.concat([data_all,data_all_for])


    # print(data_all)

    data_all.to_csv('data_all.csv', index=False)