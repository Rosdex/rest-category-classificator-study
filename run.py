# -*- encoding: utf-8 -*-
import os
import uuid
import requests
import json

from flask import Flask, request, send_file,  jsonify
from flask_restplus import Resource, Api
from werkzeug.datastructures import FileStorage
from threading import Thread

from settings import BaseConfig
from job_model import JobStatus, Job, JobSchema

# Extensions initialization
# =========================
app = Flask(__name__)
api = Api(app, version='1.0', title='Name classificator study API',
    description='This service using for training name classification model',
)

ns = api.namespace('new-jobs', description='Jobs for ML models study')

upload_parser = api.parser()
upload_parser.add_argument('file', location=BaseConfig.UPLOAD_DIR,
                           type=FileStorage, required=True)

# Data layer - Collections
# ========================
jobs = []

# Routes
# ======
@ns.route('/')
class JobList(Resource):
    # Shows a list of all jobs, and lets you POST to add new jobs
    def get(self):
        # List all jobs
        schema = JobSchema(many=True)
        jobs_dump = schema.dump(jobs)
        return jsonify(jobs_dump.data)

    @ns.expect(upload_parser)
    def post(self):
        # Create a new job
        # Step 0 - Generate UUID for new Job
        job_uuid = str(uuid.uuid1())

        # Step 1 - upload file
        input_filename = upload_file(job_uuid)
        print('File was uploaded, name is {0}'.format(input_filename))

        # Step 2 - Create job
        job = Job(job_uuid, input_filename)

        # Step 3 - Save data about Job
        jobs.append(job)

        # Step 4 - Send response 
        return job_to_json(job)

@ns.route('/<uuid>')
@ns.param('uuid', 'The job identifier')
class JobView(Resource):
    # Show a single job 
    def get(self, uuid):
        job = get_job_by_id(uuid)
        if job:
            return job_to_json(job)
        else:
            return "Job not found", 404

@ns.route('/<uuid>/perform')
@ns.param('uuid', 'The job identifier')
class JobPerformaer(Resource):
    # Perform specific Job
    def post(self, uuid):
        job = get_job_by_id(uuid)
        if job:
            thr = Thread(target=perform_async_job, args=[app, job])
            thr.start()
            return job_to_json(job)
        else:
            return "Job not found", 404

@ns.route('/<uuid>/public-model')
@ns.param('uuid', 'The job identifier')
class JobPublisher(Resource):
    # Public trained ML model
    def post(self, uuid):
        job = get_job_by_id(uuid)
        if job:
            if job.is_done():
                vectoriz_filename = '\\'.join([BaseConfig.RESULT_DIR, job.get_vectorizator_file()])
                classif_filename = '\\'.join([BaseConfig.RESULT_DIR, job.get_classificator_file()]) 
                url = BaseConfig.PUBLIC_MODEL_URL

                multiple_files = [
                    ('file', ('svm.sav', open(classif_filename, 'rb'), 'application/octet-stream')),
                    ('file', ('vectorizator.sav', open(vectoriz_filename, 'rb'), 'application/octet-stream'))]
                r = requests.post(url, files=multiple_files)
                return r.text, 204
            else:
                return 'Job not finished yet', 202
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