# -*- encoding: utf-8 -*-
import os
import uuid
import requests
import json

from flask import Flask, request, send_file,  jsonify
from threading import Thread

from settings import BaseConfig
from job_model import JobStatus, Job, JobSchema

# Extensions initialization
# =========================
app = Flask(__name__)

# Data layer - Collections
# ========================
jobs = []

# Routes
# ======
@app.route("/")
def defalut_route():
    return 'Welcome to catecory prediction study service'

@app.route('/jobs')
def get_jobs():
    schema = JobSchema(many=True)
    jobs_dump = schema.dump(jobs)
    return jsonify(jobs_dump.data)

@app.route('/jobs', methods=['POST'])
def create_job():
    # Step 0 - Generate UUID for new Job
    job_uuid = uuid.uuid1().hex

    # Step 1 - upload file
    input_filename = upload_file(job_uuid)
    print('File was uploaded, name is {0}'.format(input_filename))

    # Step 2 - Create job
    job = Job(job_uuid, input_filename)

    # Step 3 - Save data about Job
    jobs.append(job)

    # Step 4 - Send response 
    return job_to_json(job)

@app.route('/jobs/<uuid>', methods = ['GET'])
def get_job(uuid):
    job = get_job_by_id(uuid)

    if job:
        return job_to_json(job)
    else:
        return "", 404

@app.route('/jobs/<uuid>/perform', methods = ['POST'])
def perform_job(uuid):
    job = get_job_by_id(uuid)

    if job:
        thr = Thread(target=perform_async_job, args=[app, job])
        thr.start()
        return job_to_json(job)
    else:
        return "", 404

@app.route('/jobs/<uuid>/public-model', methods = ['POST'])
def public_model(uuid):
    job = get_job_by_id(uuid)

    if job:
        vectoriz_filename = '\\'.join([BaseConfig.RESULT_DIR, job.get_vectorizator_file()])
        classif_filename = '\\'.join([BaseConfig.RESULT_DIR, job.get_classificator_file()]) 
        url = PUBLIC_MODEL_URL

        multiple_files = [
            ('file', ('svm.sav', open(classif_filename, 'rb'), 'application/octet-stream')),
            ('file', ('vectorizator.sav', open(vectoriz_filename, 'rb'), 'application/octet-stream'))]
        r = requests.post(url, files=multiple_files)
        return r.text, 204
    else:
        return "", 404

# Helper functions
# ================
def get_job_by_id(uuid):
    for job in jobs:
        if job.get_id() == uuid:
            return job
    return None

def upload_file(name_prefix):
    target = os.path.join(BaseConfig.APP_ROOT, BaseConfig.UPLOAD_DIR)

    for upload in request.files.getlist("file"):
        filename = '_'.join([name_prefix, upload.filename])
        upload.save('/'.join([target, filename]))

    return filename

def perform_async_job(app, job):
    with app.app_context():
        print('----- Start async Job -----')
        job.exec_job()
        print('----- End async Job -----')

def job_to_json(job):
    schema = JobSchema(many=False)
    job_dump = schema.dump(job)
    return jsonify(job_dump.data)