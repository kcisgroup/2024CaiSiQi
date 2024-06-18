# -*- coding: utf-8 -*-
import json
import pandas as pd
from flask import Flask, render_template, request, json, jsonify
from common.utils.rights import permission_required, view_logging_required
from models import RoleModel,FdModels, FsModels, FsuploadModel
from . import index_bp
from common.utils.rights import authorize
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)

# 一、语素相似度
def chasim(A,B):
    # 计算A和B中相同语素的个数
    samecha = 0
    for i in A:
        for j in B:
            if(i==j):
                samecha = samecha + 1
    return 2*(samecha/(len(A)+len(B)))

# 二、字序相似度：如果两个词语中相同字的前后顺序也相同，这两个词的相似度就越大
# 求在A，B中仅出现一次的语素的集合
def oncec(A,B):
    help = [val for val in A if val in B]  # 求A、B中的公共元素
    # oncec = 0
    for word in help:  # 检查所有的公共元素
        count1 = 0
        for i in A:  # 在A中检查word出现的次数
            if word == i:
                count1 = count1 + 1
        if (count1 != 1):
            help.remove(word)
            continue
        count2 = 0
        for j in B:  # 在B中检查word出现的次数
            if word == j:
                count2 = count2 + 1
        if (count2 != 1):
            help.remove(word)
            continue
    return help

# Psecond(A,B)中各相邻分量的逆序数：Psecond是根据Pfirst生成的；
def revord(A,B):
    help=oncec(A,B) # 求公共元素数组
    # 生成逆序列表步骤1
    # pfirst表示oncec中的语素在A中的位置序号，pfirst是一个向量
    pfirst=[]
    for k in help:
        for kk in range(len(A)):
            if(A[kk:kk+1] == k):
                pfirst.append(k)
    # 生成逆序列表步骤2
    # psecond(A，B)表示pfirst(A，B)中的分量按对应语素在B中的次序排序生成的向量(B中的元素在pfirst中的下标，记为psecond)
    psecond=[]
    for u in pfirst: # u为下标
        for uu in range(len(B)): #遍历B中的所有元素,找到u对应的下标uu
            if(B[uu:uu+1] == u):
                psecond.append(uu)
    # 求逆序的个数
    revord_count = 0
    for i in range(1,len(psecond)):
        if(psecond[i-1:i] > psecond[i:i+1]): #如果前一个数大于后一个数
            revord_count = revord_count+1
    return revord_count

def ordsim(A,B):
    help = oncec(A,B)
    len_oncec = len(help)
    revord_count = revord(A,B)
    if(len_oncec>1):
        return (1-(revord_count/(len_oncec-1)))
    elif(len_oncec==1):
        return 1
    elif(len_oncec==0):
        return 0

# 三、词长相似度
def lensim(A,B):
    return 1-abs((len(A)-len(B))/(len(A)+len(B)))

# 主要函数：直接调用wordsim(A,B)
def wordsim(A,B):
    a=chasim(A,B)
    b=ordsim(A,B)
    c=lensim(A,B)
    return round(0.7*a+0.29*b+0.01*c,3)

@index_bp.get('/asset/info')
# @app.route('/asset/info')
@view_logging_required  # 在访问视图函数之前进行日志记录或执行一些特定的操作
@permission_required("asset:info")
def pt_info():
    return render_template('admin/pt/pt.html')  # 返回的html视图


@app.route('/changeInfo')
@view_logging_required  # 在访问视图函数之前进行日志记录或执行一些特定的操作
def changeinfo():
    asset_code = request.args.get('asset_code', 0)
    name = request.args.get('name', 0)
    num = request.args.get('num', 0)
    lat = request.args.get('lat', 0)
    lng = request.args.get('lng', 0)
    storeadd = request.args.get('storeadd')
    description = request.args.get('description', 0)
    print(id)
    print('a')
    with open("../../static/js/markdata.json", 'r', encoding='utf-8') as fw:
        injson = json.load(fw)  # type is dict

    for i in range(len(injson['image_1']) - 1):
        if asset_code == injson['image_1'][str(i)]['资产编码']:
            injson['image_1'][str(i)]['设备名称'] = name
            injson['image_1'][str(i)]['同类型总数量'] = num
            injson['image_1'][str(i)]['其他描述/取值范围'] = description
            injson['image_1'][str(i)]['存放安装地点'] = storeadd
            injson['image_1'][str(i)]['coords']['long'] = lng
            injson['image_1'][str(i)]['coords']['lat'] = lat
    print(injson)
    print('b')

    with open("../../static/js/markdata.json", 'w', encoding='utf-8') as fw:
        json.dump(injson, fw, indent=4, ensure_ascii=False)

    with open("../../static/js/markdata.json", 'r', encoding='utf-8') as fw:
        changedjson = json.load(fw)
    print(changedjson)
    print('c')
    return jsonify(1)


user_click_node = []
@app.route('/select')
def select():
    print('zhixing')
    name = request.args.get('name')
    parent = request.args.get('parent')
    pparent = request.args.get('pparent')
    item_names = name.split(" ")
    item_name = item_names[0]
    user_click_node.append(item_name)
    print(item_name)  # 点击的节点的名称
    print('csq')
    print(user_click_node)

    # CSQ1:将用户匹配点击的结点通过文本匹配算法，输出所有可能的匹配项
    # 1、读取建设清单中所有资产的：资产编码、资产名称
    construct_list_info = dict()
    df = pd.read_excel("../../static/js/c数据库文件.xlsx")
    print(df)
    for i in range(df.shape[0]):  # 遍历所有的行
        construct_list_info[df.iloc[i, 3]] = df.iloc[i, 4]
    # 2、将user_click_node与construct_list_info中的信息进行文本匹配
    yes_matched = dict()  # 记录建设清单中已匹配的设备信息，用于最后结果的保存
    flag = len(construct_list_info) * [0]  # 用于标记建设清单中的资产项是否被匹配
    for name1 in user_click_node:  # 遍历所有的用户点击的结点名称
        max_name = ''  # 匹配到的设备名称
        max_value = 0  # 符合匹配条件的相似度
        max_num = 0  # 建设清单中匹配到的设备的excel行标
        for i in range(len(construct_list_info)):  # 遍历建设清单
            if (flag[i] == 0):  # 表示尚未匹配
                name2 = construct_list_info[df.iloc[i, 3]]  # 获取该资产的设备名称
                similarity = wordsim(name1, name2)
                # 表明是匹配的，且不会存在最大值的问题
                if (similarity == 1):
                    yes_matched[df.iloc[i, 3]] = df.iloc[i, 4]
                    # print(yes_matched)
                    flag[i] = 1
                if (similarity > max_value):
                    max_value = similarity
                    max_name = name2
                    max_num = i
            else:
                continue

    print('-----------------------------------------------------------------------------------')
    print('用户点击的结点为:{}，匹配到的建设清单中的资产项有:\n{}'.format(user_click_node, yes_matched))

    # CSQ2:将yes_matched中的资产信息写入json文件：choose_data.json
    # 将数据文件中的yes_matched挑选出来，生成对应的json文件
    json_code = list(yes_matched.keys())  # 待选择的所有资产的编码
    print(json_code)
    # 用于存放用户选中的数据
    attribute = ['存放安装地点', '一级', '二级', '资产编码', '设备名称', '同类型总数量', '其他描述/取值范围', '单位', '原值']
    df = pd.DataFrame(data=0, index=range(len(json_code)), columns=attribute)  # 通过指定的属性初始化一个df
    # 完善df的内容
    df1 = pd.read_excel("../../static/js/c数据库文件.xlsx")  # 参照文件
    for i in range(df.shape[0]):
        df.iloc[i, 3] = list(json_code)[i]
    for i in range(df1.shape[0]):  # 遍历所有的行
        for j in range(df.shape[0]):
            if (df.iloc[j, 3] == df1.iloc[i, 3]):
                df.iloc[j, 0] = df1.iloc[i, 0]
                df.iloc[j, 1] = df1.iloc[i, 1]
                df.iloc[j, 2] = df1.iloc[i, 2]
                df.iloc[j, 4] = df1.iloc[i, 4]
                df.iloc[j, 5] = df1.iloc[i, 5]
                df.iloc[j, 6] = df1.iloc[i, 6]
                df.iloc[j, 7] = df1.iloc[i, 7]
                df.iloc[j, 8] = df1.iloc[i, 8]
            else:
                continue
    print(df)
    # df1——>对应的csv文件——>对应的json文件
    df.to_csv('../../static/js/choose_data.csv', index=False)
    f = open("../../static/js/choose_data.csv", "r", encoding='utf-8')
    ls = []  # 用于存放f的内容
    for line in f:  # 将文件内容添加至列表ls中
        line = line.replace("\n", "")
        ls.append(line.split(","))  # 列表中每项之间用","隔开
    f.close()
    # 对读取的内容ls做相关处理
    fw = open("../../static/js/choose_data.json", "w", encoding='utf-8')
    for i in range(1, len(ls)):
        ls[i] = dict(zip(ls[0], ls[i]))  # 将列表内容完善为字典模式
    a = json.dumps(ls[1:], sort_keys=False, indent=4, ensure_ascii=False)
    fw.write(a)
    fw.close()

    # CSQ3:获取用户勾选的信息，显示在地图中
    # 如果结合具体的应用，此处要结合具体情况再实现一个文本匹配算法

    return jsonify(1)


@app.route('/mark')
def Marked():
    markdata = []
    values = request.args.get('values')
    values = json.loads(values)  # str转成list
    with open("../../static/js/choose_data.json", 'r', encoding='utf-8') as fw:
        choosedata = json.load(fw)
    with open("../../static/js/data.json", 'r', encoding='utf-8') as fw:
        desdata = json.load(fw)

    # 将两个文件的数据合并，生成markchoosedata
    for i in range(len(values)):
        for j in range(len(choosedata)):
            if values[i] == choosedata[j]['资产编码']:
                for k in range(len(desdata['image_1']) - 1):
                    if values[i] == desdata['image_1'][str(k)]['资产编码']:
                        markchoosedata = {"资产编码": choosedata[j]['资产编码'],
                                          "设备名称": choosedata[j]['设备名称'],
                                          "同类型总数量": choosedata[j]['同类型总数量'],
                                          "其他描述/取值范围": choosedata[j]['其他描述/取值范围'],
                                          "存放安装地点": choosedata[j]['存放安装地点'],
                                          "一级": choosedata[j]['一级'],
                                          "二级": choosedata[j]['二级'],
                                          "coords": {"lat": desdata['image_1'][str(k)]['latitude'],
                                                     "long": desdata['image_1'][str(k)]['longitude']}}
                        markdata.append(markchoosedata)

    print("1",markdata)

    markdata1 = markdata.copy()
    markdatas={'image_1':{}}
    for j in range(len(markdata1)):
        m = str(j)
        markdatas['image_1'][m] = markdata1[j]

    canvas={
        "height": "2307",
        "src": "../../static/img/201中心站1.jpg",
        "width": "3047"
    }
    markdatas['image_1']['canvas']=canvas

    print(markdata1)
    with open("../../static/js/markdata.json", 'w', encoding='utf-8') as fw:
        json.dump(markdatas, fw, indent=4, ensure_ascii=False)

    return jsonify(markdata1)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
