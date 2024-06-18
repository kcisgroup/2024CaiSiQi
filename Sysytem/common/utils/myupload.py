import os

from flask import current_app

from common.flask_uploads import UploadSet, IMAGES, DOCUMENTS
from extensions import db
from models import FsuploadModel

fsuploads = UploadSet('fsuploads', 'xlsx')


# def upload_one(fsupload, mime):
def upload_one(fsupload):
    filename = fsuploads.save(fsupload)
    file_url = fsuploads.url(filename)
    upload_url = current_app.config.get("UPLOADED_FSUPLOADS_DEST")
    size = os.path.getsize(upload_url + '/' + filename)
    # fsupload = FsuploadModel(name=filename, mime=mime, size=size)
    fsupload = FsuploadModel(name=filename, size=size)
    db.session.add(fsupload)
    db.session.commit()
    return file_url


def delete_fsupload_by_id(_id):
    fsupload_name = FsuploadModel.query.filter_by(id=_id).first().name
    fsupload = FsuploadModel.query.filter_by(id=_id).delete()
    db.session.commit()
    upload_url = current_app.config.get("UPLOADED_FSUPLOADS_DEST")
    os.remove(upload_url + '/' + fsupload_name)
    return fsupload
