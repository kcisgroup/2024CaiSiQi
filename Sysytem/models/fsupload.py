from datetime import datetime

from extensions import db


class FsuploadModel(db.Model):
    __tablename__ = 'fsupload_fsupload' # 存放上传图片表格的名称
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.CHAR(250), nullable=False)
    # mime = db.Column(db.CHAR(500), nullable=False)
    size = db.Column(db.CHAR(30), nullable=False)

    create_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')
    update_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')
