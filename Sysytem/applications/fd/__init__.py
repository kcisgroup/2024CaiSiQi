from common import register_api
from flask_restful import Api
# from .fd import FdResource, FdResources
from .fd import FdResource, FdResources


def register_fd_api(api_bp):
    fd_api = Api(api_bp, prefix='/fd')
    fd_api.add_resource(FdResource, '/fd')
    fd_api.add_resource(FdResources, '/fd/<int:fd_id>')