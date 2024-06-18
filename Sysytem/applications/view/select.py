# -*- coding: utf-8 -*-
import json
import pandas as pd
from flask import Flask, render_template, request, json, jsonify, make_response
from common.utils.rights import permission_required, view_logging_required
from models import RoleModel,FdModels, FsModels, FsuploadModel
from . import index_bp
from common.utils.rights import authorize
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['LANG'] = 'en_US.UTF-8'


@index_bp.get('/select/info')
@view_logging_required
def select_info():
    return render_template('admin/select/select.html')


