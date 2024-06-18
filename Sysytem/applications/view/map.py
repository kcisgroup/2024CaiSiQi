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

# app = Flask(__name__)
# CORS(app)

@index_bp.get('/map/info')
@view_logging_required
def map_info():
    return render_template('admin/map/map.html')