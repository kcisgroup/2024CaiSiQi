# 词语相似度wordsim(A,B) = α*chasim(A,B) + β*ordsim(A,B) + γ*lensim(A,B)

# 1)语素相似度
def chasim(A,B):
    samecha = 0
    for i in A:
        for j in B:
            if(i==j):
                samecha = samecha + 1
    return 2*(samecha/(len(A)+len(B)))

# 2)字序相似度
def oncec(A,B):
    help = [val for val in A if val in B]  # 求A、B中的公共元素
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

# 3) 词长相似度
def lensim(A,B):
    return 1-abs((len(A)-len(B))/(len(A)+len(B)))

def wordsim(A,B):
    a=chasim(A,B)
    b=ordsim(A,B)
    c=lensim(A,B)
    return round(0.7*a+0.29*b+0.01*c,3)
