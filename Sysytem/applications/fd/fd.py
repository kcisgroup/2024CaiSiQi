from flask import request, jsonify
from flask_restful import Resource, reqparse

from common.utils.http import fail_api, success_api,table_api
from extensions import db
from models import FdModels, FsModels, FsuploadModel
from sqlalchemy import desc
from flask.views import MethodView
from common.utils.myupload import upload_one, delete_fsupload_by_id


class FdResource(MethodView):
    def get(self, fsupload_id=None):
        if fsupload_id is None:
            page = request.args.get('page', type=int, default=1)
            limit = request.args.get('limit', type=int, default=10)
            fsupload_paginate = FsuploadModel.query.order_by(
                desc(FsuploadModel.create_at)).paginate(
                page=page, per_page=limit, error_out=False)
            data = [
                {
                    'id': item.id,
                    'name': item.name,
                    'size': str(int(item.size)//1024)+' KB',
                    'ext': item.ext if hasattr(item, 'ext') else "",
                    'create_at': str(item.create_at),
                } for item in fsupload_paginate.items
            ]
            return table_api(
                result={
                    'items': data,
                    'total': fsupload_paginate.total,
                },
                code=0)
        else:
            item = FsuploadModel.query.get(fsupload_id)
            return table_api(
                result={
                    'items': {
                        'id': item.id,
                        'name': item.name,
                        'size': str(int(item.size)//1024)+' KB',
                        'ext': item.ext if hasattr(item, 'ext') else "",
                        'create_at': str(item.create_at),
                    },
                    'total': 1,
                },
                code=0
            )

    def post(self):
        if 'file' in request.files:
            fsupload = request.files['file']
            file_url = upload_one(fsupload=fsupload)

            res = {
                "message": "上传成功",
                "code": 0,
                "success": True,
                "data": {"src": file_url},
            }
            return jsonify(res)
        return fail_api()


class FdResources(Resource):
    def delete(self, fsupload_id):
        res = delete_fsupload_by_id(fsupload_id)
        if res:
            return success_api(message="删除成功")
        else:
            return fail_api(message="删除失败")

    def put(self, fsupload_id):
        pass
