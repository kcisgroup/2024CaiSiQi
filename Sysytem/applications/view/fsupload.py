from flask import render_template

from applications.view import index_bp
from common.utils.rights import view_logging_required, permission_required



@index_bp.get('/fsupload')
@view_logging_required
@permission_required("admin:fsupload:main")
def fsupload_index():
    return render_template('admin/fsupload/fsupload.html')


@index_bp.get('/fsupload/fsupload_add')
@view_logging_required
@permission_required("admin:fsupload:main")
def fsupload_fsupload_add():
    return render_template('admin/fsupload/fsupload_add.html')