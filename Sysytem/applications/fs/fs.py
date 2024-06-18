from flask import request, jsonify
from flask_restful import Resource, reqparse

from common.utils.http import fail_api, success_api,table_api
from extensions import db
from models import FsModels,UserModel, FsuploadModel
from sqlalchemy import desc


class FsResource(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('page',type=int, default = 1)
        parser.add_argument('limit',type=int, default = 10)
        parser.add_argument('name',type=str, default = '')

        res = parser.parse_args()

        filters = []
        if res.name:
            filters.append(FsModels.name.like('%'+res.name+'%'))
        paginate = FsModels.query.filter(*filters).paginate(page=res.page, per_page=res.limit, error_out=False)

        fs_data = [{
            'id':item.id,
            'name':item.name,
            'all_num':item.all_num,
            'fraud_num':item.fraud_num,
            'nonfraud_num': item.nonfraud_num,
            'create_at': str(item.create_at),
        } for item in paginate.items]
        return table_api(result={'items':fs_data,
                                 'total':paginate.total},
                         code=0)

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("name", type=str)
        parser.add_argument("all_num", type=int, required=True, help='cannot be null!')
        parser.add_argument("fraud_num", type=int, required=True, help='cannot be null!')
        parser.add_argument("nonfraud_num", type=int, required=True, help='cannot be ull!')

        res = parser.parse_args()

        finsta = FsModels()
        finsta.name = res.name
        finsta.all_num = res.all_num
        finsta.fraud_num = res.fraud_num
        finsta.nonfraud_num = res.nonfraud_num

        db.session.add(finsta)
        db.session.commit()
        return success_api(message='add successfully!', code=0)

class FsResources(Resource):
    def delete(self,fs_id):
        finsta = FsModels.query.get(fs_id)
        db.session.delete(finsta)
        db.session.commit()
        return success_api(message="delete successfully!", code=0)

    def put(self, fs_id):
        parser = reqparse.RequestParser()
        parser.add_argument("name", type=str)
        parser.add_argument("all_num", type=int, required=True, help='cannot be null!')
        parser.add_argument("fraud_num", type=int, required=True, help='cannot be null!')
        parser.add_argument("nonfraud_num", type=int, required=True, help='cannot be ull!')

        res = parser.parse_args()

        finsta = FsModels.query.get(fs_id)
        finsta.name = res.name
        finsta.all_num = res.all_num
        finsta.fraud_num = res.fraud_num
        finsta.nonfraud_num = res.nonfraud_num
        db.session.commit()
        return success_api(message="edit successfully!", code=0)