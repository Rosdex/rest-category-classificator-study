import datetime as dt
import csv

from classificator_study_module import CategoryClassificatorStudy

from marshmallow import Schema, fields
from enum import Enum

UPLOAD_DIR = 'uploads'
RESULT_DIR = 'results'

class JobStatus(Enum):
    CREATED = "CREATED"
    PERFORMING = "PERFORMING"
    DONE = "DONE"
    ERROR = "ERROR"

class Job():
    def __init__(self, uuid, input_fiename):
        print('Create job with id = {0}'.format(uuid))
        print('Inpur filename = {0}'.format(input_fiename))

        self.uuid = uuid
        self.input_file = input_fiename
        self.output_model_file = ''
        self.output_vectorizator_file = ''
        self.status = JobStatus.CREATED
        self.created_at = dt.datetime.now()

    def exec_job(self):
        # Create Study module
        study_module = CategoryClassificatorStudy()

        # Train classificator
        study_module.train_classificator(self.input_file)

        # Save model
        self.output_vectorizator_file, self.output_model_file = study_module.save_model_files(self.uuid)

        self.status = JobStatus.DONE

    def get_id(self):
        return self.uuid

    def get_vectorizator_file(self):
        return self.output_vectorizator_file

    def get_classificator_file(self):
        return self.output_model_file

    def check_status(self):
        result = False

        if self.status != JobStatus.DONE:
            result = True

        return result

    def __repr__(self):
        return '<Job(name={self.uuid!r})>'.format(self=self)

class JobSchema(Schema):
    uuid = fields.Str()
    input_file = fields.Str()
    output_model_file = fields.Str()
    output_vectorizator_file = fields.Str()
    status = fields.Str()
    created_at = fields.Date()
