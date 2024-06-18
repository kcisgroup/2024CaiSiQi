from common import register_api
from flask_restful import Api
from .pt import PtResource


def register_pt_api(api_bp):
    pt_api = Api(api_bp, prefix='/pt')
    pt_api.add_resource(PtResource, '/pt')