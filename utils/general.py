import httpx
import asyncio
from fastapi.security import OAuth2
from fastapi.exceptions import HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def request(self, endpoint: str, request_type: str = "GET", params: dict = {}, json: dict = {}):
        """
        Send request to given endpoint
        :param endpoint:
        :param request_type:
        :param params:
        :param json:
        :return:
        """
        if request_type.upper() == "GET":
            return await self.client.get(endpoint, params=params)
        if request_type.upper() == "POST":
            return await self.client.post(endpoint, json=json)


def clean_employee_data(data: dict):
    """
    Input example:
    {
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

    Output example:

     {
        "id": "1338",
        "name": "Emp2",
        "email": "prabhashankar@qburst.com"
        "projects": [
            Mercedes-Benz VLD POC
        ],
        "managers": [
            {"name": "Deen Edger", "email": "deen@qburst.com"}
        ]
    }
    """
    employee_details = data.get("employee_details")[0]
    project_details = data.get("project_details")
    emp_id = employee_details.get("empNo")
    name = employee_details.get("firstName") + employee_details.get("lastName", "")
    email = employee_details.get("companyEmail")
    projects = list(map(lambda x: x.get("project"), project_details))
    managers = []
    for proj in project_details:
        for manager in proj["activeProjectManagers"]:
            if manager not in managers:
                managers.append(manager)

    return {
        "id": emp_id,
        "name": name,
        "email": email,
        "projects": projects,
        "managers": managers

    }


class CustomSecurity:
    def __init__(self):
        self.client = Client('https://www.googleapis.com/oauth2/v1')

    @staticmethod
    def validate_token(data):
        # email = data.get('email')
        # if email:
        #     if email.partition('@')[2] == 'qburst.com':
        #         return True
        #     return False
        # return False
        return True

    async def __call__(self, request: Request):
        authorization: str = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        #         The 'param' contains auth token from app. Decode token to get user details.
        #         Verify the request is from QBurst domain.
        endpoint = '/tokeninfo'
        res = await asyncio.gather(self.client.request(endpoint=endpoint, params={
            'access_token': param
        }))
        data = res[0].json()
        validation = self.validate_token(data)
        if not validation:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return data
