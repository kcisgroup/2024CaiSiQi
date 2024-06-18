import random
from flask import render_template, Blueprint, request, Flask
from imblearn.over_sampling import SMOTE
from common.utils.rights import permission_required, view_logging_required
from models import RoleModel,FdModels, FsModels, FsuploadModel
from . import index_bp
from flask_cors import CORS
import numpy as np
from rulefit import RuleFit
# from rulelist import RuleList
from skrules import SkopeRules
import warnings
warnings.filterwarnings("ignore")


@index_bp.get('/fd/info')
@view_logging_required
@permission_required("fd:info")
def fd_info():
    return render_template('admin/fd/fd.html')


@index_bp.get('/fd/fd_models/<user_id>')
@view_logging_required
def fd_models_view(user_id):
    userhh = FdModels.query.get(user_id)
    return render_template('admin/fd/fd_models.html', user=userhh)


admin_evaluation = Blueprint('fd_models', __name__)
app = Flask(__name__)
CORS(app)


from sklearn.metrics import roc_auc_score
@index_bp.get('/fd/fd_models/submit_form0')
@view_logging_required
def submit_form0():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    final_threshold = 0.5100000000000002
    rf1 = RuleFit()
    rf1.fit(X_train, y_train)
    y_test_predict = rf1.predict(X_test)
    y_test_flag = np.where(y_test_predict > final_threshold, 1, 0)
    # accuracy/auc/specificity/precision/recall/f1score
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp+tn)/(tp+tn+fp+fn),4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'),4)
    spe = round(tn/(tn+fp),4)
    pre = round(tp/(tp+fp),4)
    rec = round(tp/(tp+fn),4)
    f1score = round((2*pre*rec)/(pre+rec),4)
    hhh = 'Accuracy:'+str(acc)+'\n'+'AUC:'+str(auc)+'\n'+'Specificity:'+str(spe)+'\n'+'Precision:'+str(pre)+'\n'+'Recall:'+str(rec)+'\n'+'F1-score:'+str(f1score)
    print(hhh)
    return hhh


@index_bp.get('/fd/fd_models/submit_form2')
@view_logging_required
def submit_form2():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    feature_names = list(df.columns)[:-1]
    clf = SkopeRules(feature_names=feature_names)
    clf.fit(X_train, y_train)
    y_test_predict = clf.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_predict).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_predict, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from interpret.glassbox import ExplainableBoostingClassifier
@index_bp.get('/fd/fd_models/submit_form3')
@view_logging_required
def submit_form3():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    ebm_classifier = ExplainableBoostingClassifier()
    ebm_classifier.fit(X_train, y_train)
    y_test_flag = ebm_classifier.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from sklearn.tree import DecisionTreeClassifier
@index_bp.get('/fd/fd_models/submit_form4')
@view_logging_required
def submit_form4():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_predict).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_predict, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:'+str(f1score)
    print(hhh)
    return hhh

from sklearn.linear_model import LogisticRegression
@index_bp.get('/fd/fd_models/submit_form5')
@view_logging_required
def submit_form5():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    clf1 = LogisticRegression(penalty='l2')
    clf1.fit(X_train, y_train)
    y_test_flag = clf1.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from sklearn.naive_bayes import GaussianNB
@index_bp.get('/fd/fd_models/submit_form6')
@view_logging_required
def submit_form6():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    gsn = GaussianNB()
    gsn.fit(X_train, y_train)
    y_test_flag = gsn.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

import xgboost as xgb
@index_bp.get('/fd/fd_models/submit_form7')
@view_logging_required
def submit_form7():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    best_xgbboost_clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.3, max_depth=7)  # 最终确定的参数
    best_xgbboost_clf.fit(X_train, y_train)
    y_test_flag = best_xgbboost_clf.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh


from catboost import CatBoostClassifier
@index_bp.get('/fd/fd_models/submit_form8')
@view_logging_required
def submit_form8():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    best_catboost_clf = CatBoostClassifier(learning_rate=0.03, iterations=500)  # 最终确定的参数
    best_catboost_clf.fit(X_train, y_train)
    y_test_flag = best_catboost_clf.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from sklearn.ensemble import AdaBoostClassifier
@index_bp.get('/fd/fd_models/submit_form9')
@view_logging_required
def submit_form9():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_test_flag = abc.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

import lightgbm as lgb
@index_bp.get('/fd/fd_models/submit_form10')
@view_logging_required
def submit_form10():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
    }
    lgb_classifier = lgb.train(params, train_data)
    y_predict = lgb_classifier.predict(X_train)
    max_accuarcy = 0
    final_threshold = 0
    for threshold in np.arange(0.3, 0.9, 0.01):
        y_predict_flag = []
        for i in range(y_predict.shape[0]):
            if (y_predict[i] < threshold):
                y_predict_flag.append(0)
            else:
                y_predict_flag.append(1)
        acc = accuracy_score(y_train, y_predict_flag)
        if (acc >= max_accuarcy):
            max_accuarcy = acc
            final_threshold = threshold
    y_predict = lgb_classifier.predict(X_test)
    y_test_flag = np.where(y_predict > final_threshold, 1, 0)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from sklearn.ensemble import BaggingClassifier, VotingClassifier
@index_bp.get('/fd/fd_models/submit_form11')
@view_logging_required
def submit_form11():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    bgc = BaggingClassifier()
    bgc.fit(X_train, y_train)
    y_test_flag = bgc.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

@index_bp.get('/fd/fd_models/submit_form12')
@view_logging_required
def submit_form12():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    clf1 = DecisionTreeClassifier()
    clf2 = GaussianNB()
    estimators = [("决策树", clf1), ("朴素贝叶斯", clf2)]
    clf = VotingClassifier(estimators, voting='soft')
    clf.fit(X_train, y_train)
    y_test_flag = clf.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from sklearn.ensemble import RandomForestClassifier
@index_bp.get('/fd/fd_models/submit_form13')
@view_logging_required
def submit_form13():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_test_flag = rfc.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh


from sklearn.svm import LinearSVC
@index_bp.get('/fd/fd_models/submit_form14')
@view_logging_required
def submit_form14():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    linearsvc = LinearSVC(C=1e9)
    linearsvc.fit(X_train, y_train)
    y_test_flag = linearsvc.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh


from sklearn.neural_network import MLPClassifier
@index_bp.get('/fd/fd_models/submit_form15')
@view_logging_required
def submit_form15():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    # print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    best_model = MLPClassifier(activation='relu', alpha=0.01, hidden_layer_sizes=(300,), learning_rate_init=0.001,
                               max_iter=1200)
    best_model.fit(X_train, y_train)
    y_test_flag = best_model.predict(X_test)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

from keras.layers import Dense, SimpleRNN
from keras import Sequential
@index_bp.get('/fd/fd_models/submit_form16')
@view_logging_required
def submit_form16():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    sequence_length = 10  # 序列长度
    input_dim = 50  # 特征维度

    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    X = np.array(sequences)
    y = y[sequence_length - 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    # 创建RNN模型
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(sequence_length, input_dim)))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    for k in range(1):
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        # 根据训练数据找阈值
        y_predict = model.predict(X_train)
        max_acc = 0
        final_yuzhi = 0
        for yuzhi in np.arange(0.3, 0.8, 0.01):
            y_predict_flag = []
            for i in y_predict:
                if (i < yuzhi):
                    y_predict_flag.append(0)
                else:
                    y_predict_flag.append(1)
            acc = accuracy_score(y_train, y_predict_flag)
            if acc >= max_acc:
                max_acc = acc
                final_yuzhi = yuzhi
        # 预测数据
        y_predict1 = model.predict(X_test)
        y_test_flag = []
        for i in y_predict1:
            if (i < final_yuzhi):
                y_test_flag.append(0)
            else:
                y_test_flag.append(1)
    tp, fp, fn, tn = confusion_matrix(y_test, y_test_flag).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(y_test, y_test_flag, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.nn import Dropout
from torch.optim import lr_scheduler
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler
@index_bp.get('/fd/fd_models/submit_form17')
@view_logging_required
def submit_form17():
    name = request.args.get('name')
    df = pd.read_excel('././static/upload/' + str(name))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float).view(-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled.numpy(), y.numpy(), test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

    # 将划分后的数据重新转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    # 创建图数据对象，添加边信息
    train_data = Data(x=X_train, edge_index=torch.tensor([[i, i] for i in range(X_train.size(0))],
                                                         dtype=torch.long).t().contiguous(), y=y_train)
    validation_data = Data(x=X_val, edge_index=torch.tensor([[i, i] for i in range(X_val.size(0))],
                                                            dtype=torch.long).t().contiguous(), y=y_val)
    test_data = Data(x=X_test, edge_index=torch.tensor([[i, i] for i in range(X_test.size(0))],
                                                       dtype=torch.long).t().contiguous(), y=y_test)

    # 定义GNN模型
    class GNNModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, dropout_prob):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.dropout1 = Dropout(p=dropout_prob)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.dropout2 = Dropout(p=dropout_prob)
            self.conv3 = GCNConv(hidden_channels, out_channels)
            self.dropout3 = Dropout(p=dropout_prob)

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.dropout1(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)
            x = self.dropout2(x)
            x = self.conv3(x, edge_index)
            x = self.dropout3(x)
            return x

    best_params = {'lr': 0.005, 'dropout_prob': 0.15, 'hidden_channels': 512, 'step_size': 18, 'gamma': 0.99}

    # 使用最优参数创建模型
    best_model = GNNModel(in_channels=50, hidden_channels=best_params['hidden_channels'], out_channels=1,
                          dropout_prob=best_params['dropout_prob'])
    best_optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
    best_scheduler = lr_scheduler.StepLR(best_optimizer, step_size=best_params['step_size'], gamma=best_params['gamma'])
    train_loader = DataLoader([train_data], batch_size=64, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 50
    for epoch in range(num_epochs):
        best_model.train()
        for batch in train_loader:
            best_optimizer.zero_grad()
            output = best_model(batch)
            loss = criterion(output.view(-1), batch.y)
            loss.backward()
            best_optimizer.step()
        best_scheduler.step()

    best_model.eval()
    with torch.no_grad():
        test_output = best_model(test_data)
        test_loss = criterion(test_output.view(-1), test_data.y)
        predictions = (torch.sigmoid(test_output) > 0.5).float()

    tp, fp, fn, tn = confusion_matrix(test_data.y, predictions).ravel()
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    auc = round(roc_auc_score(test_data.y, predictions, average='macro'), 4)
    spe = round(tn / (tn + fp), 4)
    pre = round(tp / (tp + fp), 4)
    rec = round(tp / (tp + fn), 4)
    f1score = round((2 * pre * rec) / (pre + rec), 4)
    hhh = 'Accuracy:' + str(acc) + '\n' + 'AUC:' + str(auc) + '\n' + 'Specificity:' + str(
        spe) + '\n' + 'Precision:' + str(pre) + '\n' + 'Recall:' + str(rec) + '\n' + 'F1-score:' + str(f1score)
    print(hhh)
    return hhh

@index_bp.get('/fd/rule_models/<user_id>')
@view_logging_required
def rule_models_view(user_id):
    userhh = FdModels.query.get(user_id)
    return render_template('admin/fd/rule_models.html', user=userhh)

@index_bp.get('/fd/rule_models/allrules')
@view_logging_required
def generate_rules():
    # name = request.args.get('name')
    # df = pd.read_excel('././static/upload/' + str(name))
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1:]
    # X = np.array(X)
    # y = np.array(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    # rf1 = RuleFit()
    # rf1.fit(X_train, y_train)
    # df_rules = rf1.get_rules()
    # return str(df_rules.iloc[0:10,0:2])

    with open('././static/upload/new_all_rules.txt','r') as f:
        all_rules = f.readlines()
    random.shuffle(all_rules)
    random_rules =  all_rules[:10]
    random_rules = ''.join(random_rules)
    return random_rules


@index_bp.get('/fd/eva_models/<user_id>')
@view_logging_required
def eva_models_view(user_id):
    userhh = FdModels.query.get(user_id)
    return render_template('admin/fd/eva_models.html', user=userhh)