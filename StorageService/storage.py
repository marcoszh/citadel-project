import os
import json
import logging

from gridfs import GridFS
from pymongo import MongoClient
from flask import Flask, make_response
from flask import request, Response

app = Flask(__name__)

MONGODB_IP = os.getenv("MONGODB_IP")

mongo_client = MongoClient(MONGODB_IP)

db = {'model': mongo_client['model'], 'gradients': mongo_client['gradients'],
      'data': mongo_client['data'], 'test': mongo_client['test'], 'mask': mongo_client['mask']}
grid_fs = {'model': GridFS(db['model']), 'gradients': GridFS(
    db['gradients']), 'data': GridFS(db['data']), 'test': GridFS(db['test']), 'mask': GridFS(db['mask'])}


@app.route('/upload_model/<file_name>', methods=['POST'])
def upload_model(file_name):
    file_id = grid_fs['model'].put(request.data, filename=file_name)

    # app.logger.error(grid_fs['model'].find_one(file_id).read())

    if grid_fs['model'].find_one(file_id) is not None:
        return json.dumps({'status': 'File saved successfully'}), 200
    else:
        return json.dumps({'status': 'Error occurred while saving file.'}), 500


@app.route('/download_model/<file_name>', methods=['GET'])
def download_model(file_name):
    grid_fs_file = grid_fs['model'].find_one(
        {'filename': file_name}, sort=[('datetime', -1)])
    return Response(response=grid_fs_file.read(), status=200,
                    mimetype="application/octet_stream")


@app.route('/upload_gradients/<file_name>', methods=['POST'])
def upload_gradients(file_name):
    with grid_fs['gradients'].new_file(filename=file_name) as fp:
        fp.write(request.data)
        file_id = fp._id

    if grid_fs['gradients'].find_one(file_id) is not None:
        return json.dumps({'status': 'File saved successfully'}), 200
    else:
        return json.dumps({'status': 'Error occurred while saving file.'}), 500


@app.route('/download_gradients/<file_name>', methods=['GET'])
def download_gradients(file_name):
    grid_fs_file = grid_fs['gradients'].find_one({'filename': file_name})
    response = make_response(grid_fs_file.read())
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers["Content-Disposition"] = "attachment; filename={}".format(
        file_name)
    return response


@app.route('/upload_data/<file_name>', methods=['POST'])
def upload_data(file_name):
    with grid_fs['data'].new_file(filename=file_name) as fp:
        fp.write(request.data)
        file_id = fp._id

    if grid_fs['data'].find_one(file_id) is not None:
        return json.dumps({'status': 'File saved successfully'}), 200
    else:
        return json.dumps({'status': 'Error occurred while saving file.'}), 500


@app.route('/download_data/<file_name>', methods=['GET'])
def download_data(file_name):
    grid_fs_file = grid_fs['data'].find_one({'filename': file_name})
    response = make_response(grid_fs_file.read())
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers["Content-Disposition"] = "attachment; filename={}".format(
        file_name)
    return response


@app.route('/upload_mask/<file_name>', methods=['POST'])
def upload_mask(file_name):
    with grid_fs['mask'].new_file(filename=file_name) as fp:
        fp.write(request.data)
        file_id = fp._id

    if grid_fs['data'].find_one(file_id) is not None:
        return json.dumps({'status': 'File saved successfully'}), 200
    else:
        return json.dumps({'status': 'Error occurred while saving file.'}), 500


@app.route('/download_mask/<file_name>', methods=['GET'])
def download_mask(file_name):
    grid_fs_file = grid_fs['mask'].find_one({'filename': file_name})
    response = make_response(grid_fs_file.read())
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers["Content-Disposition"] = "attachment; filename={}".format(
        file_name)
    return response


def clean_up():
    for key, gfs_c in grid_fs.items:
        for file_item in gfs_c.find():
            gfs_c.delete(file_item._id)


if __name__ == "__main__":
    app.run(debug=False, port=80, host='0.0.0.0',
            threaded=False, processes=4)
    app.logger.setLevel(logging.INFO)
    # a = grid_fs['test'].put(b"fghfuweguwew", filename="foo")
    # print(grid_fs['test'].find_one({'filename':'foo'}).read())