from flask import render_template
from flask_login import login_required, current_user
from sqlalchemy import desc

from common.utils.rights import permission_required, view_logging_required
from models import LogModel, RoleModel, UserModel,FsModels
from . import index_bp


@index_bp.get('/fs/info')
@view_logging_required
@permission_required("fs:info")
def fs_info():
    return render_template('admin/fs/fs.html')

@index_bp.get('/fs/add')
@view_logging_required
def fs_add_view():
    roles = RoleModel.query.all()
    return render_template('admin/fs/fs_add.html', roles=roles)

@index_bp.get('/fs/<user_id>')
@view_logging_required
def fs_fs_id_view(user_id):
    userhh = FsModels.query.get(user_id)
    return render_template('admin/fs/fs_edit.html', user=userhh)
