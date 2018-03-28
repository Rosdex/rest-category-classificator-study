import os
import uuid
import requests
import json

#from classificator import CategoryClassificator
from job_model import JobStatus, Job, JobSchema
from flask import Flask, request, send_file,  jsonify

# Constants
# =========
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = 'uploads'
RESULT_DIR = 'results'

# Extensions initialization
# =========================
app = Flask(__name__)

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

    # Step 4 - Prepare response 
    schema = JobSchema(many=False)
    job_dump = schema.dump(job)

    return jsonify(job_dump.data)

@app.route('/jobs/<uuid>', methods = ['GET'])
def get_job(uuid):
    job = get_job_by_id(uuid)

    if job != None:
        schema = JobSchema(many=False)
        job_dump = schema.dump(job)
        return jsonify(job_dump.data)
    else:
        return "", 404

@app.route('/jobs/<uuid>/perform', methods = ['POST'])
def perform_job(uuid):
    job = get_job_by_id(uuid)

    if job != None:
        job.exec_job()
        return "", 204
    else:
        return "", 404

@app.route('/jobs/<uuid>/public-model', methods = ['POST'])
def public_model(uuid):
    job = get_job_by_id(uuid)

    if job != None:
        vectoriz_filename = '\\'.join([RESULT_DIR, job.get_vectorizator_file()])
        classif_filename = '\\'.join([RESULT_DIR, job.get_classificator_file()]) 

        url = 'http://127.0.0.1:5001/ml-models'

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
    target_job = None

    for job in jobs:
        if job.get_id() == uuid:
            target_job = job

    return target_job

def upload_file(name_prefix):
    target = os.path.join(APP_ROOT, UPLOAD_DIR)

    for upload in request.files.getlist("file"):
        filename = '_'.join([name_prefix, upload.filename])
        upload.save('/'.join([target, filename]))

    return filename
