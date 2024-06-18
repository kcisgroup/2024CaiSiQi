from flask import request, jsonify
from flask.views import MethodView
from sqlalchemy import desc

from common.utils.http import fail_api, success_api, table_api
from models import MatchModels

class MatchAPI(MethodView):

    def get(self, match_id):
        if match_id is None:
            page = request.args.get('page', type=int, default=1)
            limit = request.args.get('limit', type=int, default=10)
            match_paginate = MatchModels.query.order_by(
                desc(MatchModels.id)).paginate(
                page=page, per_page=limit, error_out=False)
            data = [
                {
                    'id': item.id,
                    'name': item.name,
                    'num': item.num,
                    'place': item.place,
                    'first': item.first,
                    'second': item.second,
                    'rangee': item.rangee,
                    'unit': item.unit,
                    'original': item.original,
                    'size': item.size,
                } for item in match_paginate.items
            ]
            return table_api(
                result={
                    'items': data,
                    'total': match_paginate.total,
                },
                code=0)
        else:
            item = MatchModels.query.get(match_id)
            return table_api(
                result={
                    'items': {
                        'id': item.id,
                        'name': item.name,
                        'num': item.num,
                        'place': item.place,
                        'first': item.first,
                        'second': item.second,
                        'rangee': item.rangee,
                        'unit': item.unit,
                        'original': item.original,
                        'size': item.size,
                    },
                    'total': 1,
                },
                code=0
            )