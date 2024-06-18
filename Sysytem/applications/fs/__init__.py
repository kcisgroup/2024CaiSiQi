from common import register_api
from flask_restful import Api
from .fs import FsResource, FsResources


def register_fs_api(api_bp):
    fs_api = Api(api_bp, prefix='/fs')
    fs_api.add_resource(FsResource, '/fs')
    fs_api.add_resource(FsResources, '/fs/<int:fs_id>')