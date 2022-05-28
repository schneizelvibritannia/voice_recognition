import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, List

import uvicorn
from pydantic import BaseModel

from fastapi import FastAPI, Body, Depends, Header, status
from dotenv import load_dotenv

from utils.nlp.ner_spacy import SpaceStatusReportNER
from utils.nlp.text_classifier import TextClassifier
from utils.nlp.qa_models import QaModel
from utils.nlp.generic import correct_sentence

from utils.messages import SUCCESS, ERROR
from utils.logging import Logger
from utils.constants import CLASSIFICATION_THRESHOLD, EMPLOYEE_DETAILS
from utils.general import Client, clean_employee_data, CustomSecurity

from urls import URLS
from schemas import *
from database import SessionLocal, metadata, database_URL, user

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

database = database_URL
db = SessionLocal()
load_dotenv()

app = FastAPI()
auth_scheme = CustomSecurity()


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


@app.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(users: UserIn):
    query = user.insert().values(username=users.username, email=users.email, designation=users.designation)
    last_record_id = await database.execute(query)
    return {**users.dict(), "id": last_record_id}


@app.get("/users/", response_model=List[User], status_code=status.HTTP_200_OK)
async def read_user(skip: int = 0, take: int = 20):
    query = user.select().offset(skip).limit(take)
    return await database.fetch_all(query)


@app.get("/users/{user_id}/", response_model=User, status_code=status.HTTP_200_OK)
async def read_users(user_id: int):
    query = user.select().where(user.c.id == user_id)
    return await database.fetch_one(query)


@app.delete("/users/{user_id}/", status_code=status.HTTP_200_OK)
async def delete_user(user_id: int):
    query = user.delete().where(user.c.id == user_id)
    await database.execute(query)
    return {"message": "User with id: {} deleted successfully!".format(user_id)}


status_ner = SpaceStatusReportNER('./models/ner/status/')
status_qa_model = QaModel('./models/qa-model')
qa_model = QaModel()
text_classifier = TextClassifier('./models/text-classifier')
client = Client(base_url=os.environ.get('HOST'))


@app.post("/classify-text-ner")
def classify_text_ner(item: InputText = Body(default={
    "input_text": 'This is a test input for text classification.'
},
    examples={
        "status_report": {
            "value": {
                "input_text": 'Yesterday I worked for 3 hours on API integration for Space project.'
            }
        },
        "vacation_request": {
            "value": {
                "input_text": 'I am on leave next friday as I have a medical appointment.'
            }
        }
    }
)):
    """
    Classify input text as status report or vacation request using NER.
    """
    try:
        input_text = item.input_text
        classification = text_classifier.process(input_text, threshold=CLASSIFICATION_THRESHOLD)
        if classification:
            Logger.get_logger().info(
                'Classified input text using NER - Spacy. Input text: {}, classification: {}.'.format(input_text,
                                                                                                      classification))
            return {
                'success': True,
                'code': 200,
                'data': classification,
                'message': SUCCESS['TEXT_CLASSIFICATION']
            }
        else:
            Logger.get_logger().info(
                'Classifying input text using NER - Spacy failed. Input text: {}.'.format(input_text))
            return {
                'success': False,
                'code': 200,
                'message': ERROR['TEXT_CLASSIFICATION']
            }

    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post("/classify-text-qa")
def classify_text_qa(item: InputText = Body(default={
    "input_text": 'This is a test input for text classification using qa model.'
},
    examples={
        "status_report": {
            "value": {
                "input_text": 'Yesterday I worked for 3 hours on API integration for Space project.'
            }
        },
        "vacation_request": {
            "value": {
                "input_text": 'I am on leave next friday as I have a medical appointment.'
            }
        }
    }
)):
    """
        Classify input text as status report or vacation request using QA model
    """
    try:
        input_text = item.input_text
        classification = qa_model.status_or_vacation(input_text)
        if classification:
            Logger.get_logger().info(
                'Classified input text using QA. Input text: {}, classification: {}.'.format(input_text,
                                                                                             classification))
            return {
                'success': True,
                'code': 200,
                'data': classification,
                'message': SUCCESS['TEXT_CLASSIFICATION']
            }
        else:
            Logger.get_logger().info('Classifying input text using QA failed. Input text: {}.'.format(input_text))
            return {
                'success': False,
                'code': 200,
                'message': ERROR['TEXT_CLASSIFICATION']
            }
    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.get('/employee-details')
def get_employee_details(employee_id: str = None):
    """
    Fetch employee details
    :param employee_id:
    :return:
    """
    try:
        if employee_id:
            employee = list(filter(lambda x: x['id'] == employee_id, EMPLOYEE_DETAILS))
            if len(employee) > 0:
                Logger.get_logger().info(
                    'Employee details fetched successfully for employee id: {}'.format(employee_id))
                return {
                    'success': True,
                    'code': 200,
                    'data': employee[0],
                    'message': SUCCESS['EMPLOYEE_DETAILS']
                }
            Logger.get_logger().info('Employee details not found.')
            return {
                'success': True,
                'code': 200,
                'data': None,
                'message': ERROR['EMPLOYEE_DETAILS_NOT_FOUND'],
            }
        Logger.get_logger().info('Employee details fetched successfully.')
        return {
            'success': True,
            'code': 200,
            'message': SUCCESS['EMPLOYEE_DETAILS'],
            'data': EMPLOYEE_DETAILS
        }

    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post('/clean-employee-details')
def clean_employee_details(data: EmployeeData):
    """
    Clean input employee detail dict.

    """
    try:
        if data:
            employee = clean_employee_data(dict(data))
            Logger.get_logger().info(
                'Employee details: {}'.format(employee))
            return {
                'success': True,
                'code': 200,
                'data': employee,
                'message': SUCCESS['EMPLOYEE_DETAILS']
            }

    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post('/status-report-ner')
def get_status_report(item: StatusInput = Body(default={
    "input_text": 'This is a test input for processing status report using ner.'
},
    examples={
        "status_report": {
            "value": {
                "input_text": 'Yesterday I worked for 3 hours on API integration for Space project.',
                "employee_details": """{ 'id': "1", 'name': 'Emp1', 'projects': [ 'RPA', 'FPSB'], 
                'managers': [{'name': 'Ameena', 'email': 'm1@qburst.com'} ]}"""
            }
        },
    }
)):
    """
    Extract status report from input text using NER
    :param item:
    :return:
    """
    try:
        text = item.input_text
        employee = item.employee_details
        projects = employee.get('projects')
        status_ = status_ner.process(text, validate=True, projects=projects)
        Logger.get_logger().info(
            'Extract status report from input text using NER. Input text: {}, data: {}.'.format(text, status_))
        return {
            'success': True,
            'code': 200,
            'message': SUCCESS['STATUS_REPORT'],
            'data': status_
        }
    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post('/status-from-text-qa')
def get_status_from_text(item: InputText):
    """
    Extract status from text using QA model
    :param item:
    :return:
    """
    try:
        text = item.input_text
        status_ = status_qa_model.get_status_from_text(text)
        Logger.get_logger().info('Extract status from input text using QA. '
                                 'Input text: {}, status: {}.'.format(text,
                                                                      status_))
        return {
            'success': True,
            'code': 200,
            'message': SUCCESS['STATUS_REPORT_QA'],
            'data': status_
        }
    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post('/vacation-request-qa')
def vacation_request_qa(item: VacationInput = Body(default={
    "input_text": 'This is a test input for processing vacation request using qa model.'
},
    examples={"vacation_request": {
        "value": {
            "input_text": 'I am on leave next friday as I have a medical appointment.',
            "employee_details": """{ 'id': "1", 'name': 'Emp1', 'projects': [ 'RPA', 'FPSB'], 'managers':
             [{'name': 'Ameena', 'email': 'm1@qburst.com'} ]}"""

        }
    }
    }
)):
    try:
        text = item.input_text
        employee = item.employee_details
        managers = employee.get('managers')
        vacation_data = qa_model.process_vacation_request(text, managers=managers)
        Logger.get_logger().info(
            'Extract vacation request from input text using QA. Input text: {}, data: {}.'.format(text, vacation_data))
        return {
            'success': True,
            'code': 200,
            'message': SUCCESS['VACATION_QA'],
            'data': vacation_data
        }
    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post('/correct-sentence')
def format_sentence(item: CorrectSentence):
    """
    Complete sentence and get a corrected sentence using input text and context sentence for reference
    :param item:
    :return:
    """
    try:
        text = item.input_text
        context = item.context
        corrected_sentence = correct_sentence(text, context)
        Logger.get_logger().info('Corrected input sentence. Input text:'
                                 ' {}, context: {}, corrected: {}'.format(text,
                                                                          context,
                                                                          corrected_sentence))
        return {
            'success': True,
            'code': 200,
            'message': SUCCESS['CORRECT_SENTENCE'],
            'data': corrected_sentence
        }
    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


@app.post('/process')
async def process_text(item: Item = Body(default={
    "input_text": 'This is a test input for deriving attributes in status or vacation request'
},
    examples={
        "status_report": {
            "value": {

                "input_text": 'Yesterday I worked for 3 hours on API integration for Space project.',
                "employee_details":
                    [
                        {"companyEmail": "prabhashankar@qburst.com",
                         "designation": "Architect",
                         "empNo": "1338",
                         "firstName": "Prabhashankar",
                         "floating_vacation": "20 Aug 2021",
                         "id": "597723",
                         "key": "ag5zfnFidXJzdC1zcGFjZXIQCxIIRW1wbG95ZWUY270kDA",
                         "lastName": "Rudran",
                         "leaves_remaining": 17.5,
                         "location": "SB273, Basement 2, Thejaswini, Thiruvananthapuram",
                         "loss_of_pay_leaves": 0.0}
                    ],
                "project_details":
                    [
                        {"activeProjectManagers": [{"email": "deen@qburst.com", "name": "Deen Edger", "id": 11621792}],
                         "allocation": 70,
                         "employee": "Prabhashankar Rudran",
                         "project": "Mercedes-Benz VLD POC",
                         "projectRole": "Developer",
                         "startDate": "2021-04-26",
                         "technology": ""}
                    ]
            }
        },
        "vacation_request": {
            "value": {
                "input_text": 'I am on leave next friday as I have a medical appointment.',
                "employee_details":
                    [
                        {"companyEmail": "prabhashankar@qburst.com",
                         "designation": "Architect",
                         "empNo": "1338",
                         "firstName": "Prabhashankar",
                         "floating_vacation": "20 Aug 2021",
                         "id": "597723",
                         "key": "ag5zfnFidXJzdC1zcGFjZXIQCxIIRW1wbG95ZWUY270kDA",
                         "lastName": "Rudran",
                         "leaves_remaining": 17.5,
                         "location": "SB273, Basement 2, Thejaswini, Thiruvananthapuram",
                         "loss_of_pay_leaves": 0.0}
                    ],
                "project_details":
                    [
                        {"activeProjectManagers": [{"email": "deen@qburst.com", "name": "Deen Edger", "id": 11621792}],
                         "allocation": 70,
                         "employee": "Prabhashankar Rudran",
                         "project": "Mercedes-Benz VLD POC",
                         "projectRole": "Developer",
                         "startDate": "2021-04-26",
                         "technology": ""}
                    ]
            }
        }
    }), user_info: str = Depends(auth_scheme)
):
    # start = time.time()
    try:
        Logger.get_logger().info('User Info: {}'.format(user_info))

        text = item.input_text
        employee_details = item.employee_details
        project_details = item.project_details

        # Get employee details and classify text
        res = await asyncio.gather(
            client.request(endpoint=URLS['GET-EMPLOYEE'], json={
                "employee_details": employee_details,
                "project_details": project_details
            }, request_type='post'),
            client.request(URLS['CLASSIFY-TEXT-NER'], json={
                "input_text": text
            }, request_type='post'),
            client.request(URLS['CLASSIFY-TEXT-QA'], json={
                "input_text": text
            }, request_type='post'),
        )
        employee_data = res[0].json().get('data')
        classification_ner = res[1].json().get('data')
        classification_qa = res[2].json().get('data')
        classification = classification_ner

        if employee_data:
            ########################################################
            # Use classification by QA if NER classification fails
            if not classification_ner:
                classification = classification_qa

            if classification['class'] == 'Status Report':
                res = await client.request(URLS['STATUS-NER'],
                                           json={'input_text': text, 'employee_details': employee_data},
                                           request_type='post')
                status_data = res.json().get('data')
                status_ = status_data.get('STATUS', None)

                ##################################################
                # Use QA if STATUS not found by NER
                if not status_:
                    res = await client.request(URLS['STATUS-QA'],
                                               json={'input_text': text}, request_type='post')
                    status_ = res.json().get('data')
                ##################################################
                # Clean status
                # res = await client.request(URLS['CORRECT-SENTENCE'],
                #                            json={'input_text': status, 'context': text}, request_type='post')
                # cleaned_status = res.json().get('data')
                # status_data['STATUS'] = cleaned_status
                status_data['STATUS'] = status_
                status_data['REQUEST_TYPE'] = 'STATUS'

                # print(time.time() - start)
                Logger.get_logger().info('Extracted status report from input text. Input: {}. Entities: {}'
                                         .format(text, json.dumps(status_data)))
                return {
                    'success': True,
                    'code': 200,
                    'message': SUCCESS['PROCESS_STATUS'],
                    'data': status_data
                }

            elif classification['class'] == 'Vacation Request':
                res = await client.request(URLS['VACATION-QA'],
                                           json={'input_text': text, 'employee_details': employee_data},
                                           request_type='post')
                vacation_data = res.json().get('data')
                vacation_data['REQUEST_TYPE'] = 'VACATION'

                # Clean vacation reason
                # reason = vacation_data.get('REASON')
                # res = await client.request(URLS['CORRECT-SENTENCE'],
                #                            json={'input_text': reason, 'context': text}, request_type='post')
                # cleaned_vacation_reason = res.json().get('data')
                # vacation_data['REASON'] = cleaned_vacation_reason
                # print(time.time() - start)
                Logger.get_logger().info('Extracted vacation request from input text.'
                                         ' Input: {}. Entities: {}'.format(text,
                                                                           json.dumps(vacation_data)))

                return {
                    'success': True,
                    'code': 200,
                    'message': SUCCESS['PROCESS_VACATION'],
                    'data': vacation_data
                }
        else:
            return {
                'success': False,
                'code': 200,
                'message': ERROR['EMPLOYEE_DETAILS_NOT_FOUND']
            }
    except Exception as e:
        Logger.get_logger().exception(e)
        return {
            'success': False,
            'code': 500,
            'message': ERROR['GENERAL']
        }


if __name__ == '__main__':
    uvicorn.run('main:app')
