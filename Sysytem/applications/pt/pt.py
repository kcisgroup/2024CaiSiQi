from flask import request, jsonify
from flask_restful import Resource, reqparse

from common.utils.http import fail_api, success_api,table_api
from extensions import db
from models import FsModels,UserModel, FsuploadModel,PtModels
from sqlalchemy import desc


class PtResource(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('page',type=int, default = 1)
        parser.add_argument('limit',type=int, default = 10)
        parser.add_argument('name',type=str, default = '')

        res = parser.parse_args()

        filters = []
        if res.name:
            filters.append(PtModels.name.like('%'+res.name+'%'))
        paginate = PtModels.query.filter(*filters).paginate(page=res.page, per_page=res.limit, error_out=False)

        pt_data = [{
            'id':item.id,
            'name':item.name,
            'total':item.total,
            'other':str(item.other),
        } for item in paginate.items]
        return table_api(result={'items':pt_data,
                                 'total':paginate.total},
                         code=0)

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("name", type=str)
        parser.add_argument("total", type=int, required=True, help='cannot be null!')
        parser.add_argument("other", type=str, required=True, help='cannot be null!')

        res = parser.parse_args()

        finsta = PtModels()
        finsta.name = res.name
        finsta.total = res.total
        finsta.other = res.other

        db.session.add(finsta)
        db.session.commit()
        return success_api(message='add successfully!', code=0)