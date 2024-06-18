# # -*- coding: utf-8 -*-
# import json
# import pandas as pd
# from flask import Flask, render_template, request, json, jsonify, make_response
# from common.utils.rights import permission_required, view_logging_required
# from models import RoleModel,FdModels, FsModels, FsuploadModel
# from . import index_bp
# from common.utils.rights import authorize
# from flask_cors import CORS
# import warnings
# warnings.filterwarnings("ignore")
# import os
# os.environ['LANG'] = 'en_US.UTF-8'
#
#
# app = Flask(__name__)
# CORS(app)
#
# def chasim(A,B):
#     samecha = 0
#     for i in A:
#         for j in B:
#             if(i==j):
#                 samecha = samecha + 1
#     return 2*(samecha/(len(A)+len(B)))
#
# def oncec(A,B):
#     help = [val for val in A if val in B]
#     for word in help:
#         count1 = 0
#         for i in A:
#             if word == i:
#                 count1 = count1 + 1
#         if (count1 != 1):
#             help.remove(word)
#             continue
#         count2 = 0
#         for j in B:
#             if word == j:
#                 count2 = count2 + 1
#         if (count2 != 1):
#             help.remove(word)
#             continue
#     return help
#
# def revord(A,B):
#     help=oncec(A,B)
#     pfirst=[]
#     for k in help:
#         for kk in range(len(A)):
#             if(A[kk:kk+1] == k):
#                 pfirst.append(k)
#     psecond=[]
#     for u in pfirst:
#         for uu in range(len(B)):
#             if(B[uu:uu+1] == u):
#                 psecond.append(uu)
#     revord_count = 0
#     for i in range(1,len(psecond)):
#         if(psecond[i-1:i] > psecond[i:i+1]):
#             revord_count = revord_count+1
#     return revord_count
#
# def ordsim(A,B):
#     help = oncec(A,B)
#     len_oncec = len(help)
#     revord_count = revord(A,B)
#     if(len_oncec>1):
#         return (1-(revord_count/(len_oncec-1)))
#     elif(len_oncec==1):
#         return 1
#     elif(len_oncec==0):
#         return 0
#
# def lensim(A,B):
#     return 1-abs((len(A)-len(B))/(len(A)+len(B)))
#
# def wordsim(A,B):
#     a=chasim(A,B)
#     b=ordsim(A,B)
#     c=lensim(A,B)
#     return round(0.7*a+0.29*b+0.01*c,3)
#
# @index_bp.get('/asset/info')
# @view_logging_required
# @permission_required("asset:info")
# def pt_info():
#     return render_template('admin/pt/pt.html')
#
#
#
# @index_bp.route('/changeInfo')
# @view_logging_required
# def changeinfo():
#     asset_code = request.args.get('asset_code', 0)
#     name = request.args.get('name', 0)
#     num = request.args.get('num', 0)
#     lat = request.args.get('lat', 0)
#     lng = request.args.get('lng', 0)
#     storeadd = request.args.get('storeadd')
#     description = request.args.get('description', 0)
#     print(id)
#     print('a')
#     with open("../../static/js/markdata.json", 'r', encoding='utf-8') as fw:
#         injson = json.load(fw)  # type is dict
#
#     for i in range(len(injson['image_1']) - 1):
#         if asset_code == injson['image_1'][str(i)]['资产编码']:
#             injson['image_1'][str(i)]['设备名称'] = name
#             injson['image_1'][str(i)]['同类型总数量'] = num
#             injson['image_1'][str(i)]['其他描述/取值范围'] = description
#             injson['image_1'][str(i)]['存放安装地点'] = storeadd
#             injson['image_1'][str(i)]['coords']['long'] = lng
#             injson['image_1'][str(i)]['coords']['lat'] = lat
#     print(injson)
#     print('b')
#
#     with open("../../static/js/markdata.json", 'w', encoding='utf-8') as fw:
#         json.dump(injson, fw, indent=4, ensure_ascii=False)
#
#     with open("../../static/js/markdata.json", 'r', encoding='utf-8') as fw:
#         changedjson = json.load(fw)
#     print(changedjson)
#     print('c')
#     print(jsonify(1))
#     return jsonify(1)
#
#
# @index_bp.route('/select',methods=['GET'])
# def select():
#     user_click_node = []
#     name = request.args.get('name')
#     item_names = name.split("+")
#     item_name = item_names[0]
#     user_click_node.append(item_name)
#
#     construct_list_info = dict()
#     df = pd.read_excel("E:\pythonProject\pear-admin-flask\static\js\c数据库文件.xlsx")
#     for i in range(df.shape[0]):
#         construct_list_info[df.iloc[i, 3]] = df.iloc[i, 4]
#
#     yes_matched = dict()
#     flag = len(construct_list_info) * [0]
#     for name1 in user_click_node:
#         max_value = 0
#         for i in range(len(construct_list_info)):
#             if (flag[i] == 0):
#                 name2 = construct_list_info[df.iloc[i, 3]]
#                 similarity = wordsim(name1, name2)
#                 if (similarity == 1):
#                     yes_matched[df.iloc[i, 3]] = df.iloc[i, 4]
#                     flag[i] = 1
#                 if (similarity > max_value):
#                     max_value = similarity
#             else:
#                 continue
#
#     # 2:将yes_matched中的资产信息写入json文件：choose_data.json
#     json_code = list(yes_matched.keys())  # 待选择的所有资产的编码
#     attribute = ['存放安装地点', '一级', '二级', '资产编码', '设备名称', '同类型总数量', '其他描述/取值范围', '单位', '原值']
#     df = pd.DataFrame(data=0, index=range(len(json_code)), columns=attribute)
#     df1 = pd.read_excel("E:\pythonProject\pear-admin-flask\static\js\c数据库文件.xlsx")
#     for i in range(df.shape[0]):
#         df.iloc[i, 3] = list(json_code)[i]
#     for i in range(df1.shape[0]):
#         for j in range(df.shape[0]):
#             if (df.iloc[j, 3] == df1.iloc[i, 3]):
#                 df.iloc[j, 0] = df1.iloc[i, 0]
#                 df.iloc[j, 1] = df1.iloc[i, 1]
#                 df.iloc[j, 2] = df1.iloc[i, 2]
#                 df.iloc[j, 4] = df1.iloc[i, 4]
#                 df.iloc[j, 5] = df1.iloc[i, 5]
#                 df.iloc[j, 6] = df1.iloc[i, 6]
#                 df.iloc[j, 7] = df1.iloc[i, 7]
#                 df.iloc[j, 8] = df1.iloc[i, 8]
#             else:
#                 continue
#
#     df.to_csv('E:\pythonProject\pear-admin-flask\static\js\choose_data.csv', index=False)
#
#     f = open("E:\pythonProject\pear-admin-flask\static\js\choose_data.csv", "r", encoding='utf-8')
#     ls = []
#     for line in f:  # 将文件内容添加至列表ls中
#         line = line.replace("\n", "")
#         ls.append(line.split(","))
#     f.close()
#
#     # with open("./static/js/choose_data.json", "a", encoding='utf-8') as f:
#     #     for i in range(1, len(ls)):
#     #         ls[i] = dict(zip(ls[0], ls[i]))
#     #     a = json.dumps(ls[1:], sort_keys=False, indent=4, ensure_ascii=False)
#     #     f.write(a)
#
#     fw = open("./static/js/choose_data.json", "w", encoding='utf-8') # 创建文件
#     for i in range(1, len(ls)):
#         ls[i] = dict(zip(ls[0], ls[i]))
#     a = json.dumps(ls[1:], sort_keys=False, indent=4, ensure_ascii=False)
#     fw.write(a)
#     fw.close()
#
#     # 3:获取用户勾选的信息，显示在地图中
#     return jsonify(1)
#
#
# @index_bp.route('/mark')
# def Marked():
#     markdata = []
#     values = request.args.get('values')
#     values = json.loads(values)  # str转成list
#     with open("E:\pythonProject\pear-admin-flask\static\js\choose_data.json", 'r', encoding='utf-8') as fw:
#         choosedata = json.load(fw)
#
#     with open("E:\pythonProject\pear-admin-flask\static\js\data.json", 'r', encoding='utf-8') as fw:
#         desdata = json.load(fw)
#
#     # 将两个文件的数据合并，生成markchoosedata
#     for i in range(len(values)):
#         for j in range(len(choosedata)):
#             if values[i] == choosedata[j]['资产编码']:
#                 for k in range(len(desdata['image_1']) - 1):
#                     if values[i] == desdata['image_1'][str(k)]['资产编码']:
#                         markchoosedata = {"资产编码": choosedata[j]['资产编码'],
#                                           "设备名称": choosedata[j]['设备名称'],
#                                           "同类型总数量": choosedata[j]['同类型总数量'],
#                                           "其他描述/取值范围": choosedata[j]['其他描述/取值范围'],
#                                           "存放安装地点": choosedata[j]['存放安装地点'],
#                                           "一级": choosedata[j]['一级'],
#                                           "二级": choosedata[j]['二级'],
#                                           "coords": {"lat": desdata['image_1'][str(k)]['latitude'],
#                                                      "long": desdata['image_1'][str(k)]['longitude']}}
#                         markdata.append(markchoosedata)
#
#     markdata1 = markdata.copy()
#     markdatas={'image_1':{}}
#     for j in range(len(markdata1)):
#         m = str(j)
#         markdatas['image_1'][m] = markdata1[j]
#
#     canvas={
#         "height": "2307",
#         "src": "../../../static/img/201中心站1.jpg",
#         "width": "3047"
#     }
#     markdatas['image_1']['canvas']=canvas
#     with open("E:\pythonProject\pear-admin-flask\static\js\markdata.json", 'w', encoding='utf-8') as fw:
#         json.dump(markdatas, fw, indent=4, ensure_ascii=False)
#
#     return jsonify(markdata1)
#
#
# if __name__ == "__main__":
#     app.run(port=5000, debug=True)
