import os
from datetime import timedelta, date
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sutime import SUTime
from fuzzywuzzy import process, fuzz
from word2number import w2n
import datetime
# New Addons
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from difflib import get_close_matches
from logging import Logger

nltk.download('punkt')
nltk.download('stopwords')


# from ner_spacy import get_match_from_list

class QaModel:
    def __init__(self, model_name="deepset/minilm-uncased-squad2"):
        """
        Class to run QA model
        :param model_name:
        """
        jar_files_dir = os.environ.get('SUTIME_PATH')

        self.model_name = model_name
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

        jar_path = None
        if jar_files_dir:
            jar_path = os.path.dirname(jar_files_dir)
        self.sutime = SUTime(jars=jar_path, mark_time_ranges=True, include_range=True)

    def get_answer(self, question, input_text):
        """
        Get answers for question on input text
        :param question:
        :param input_text:
        :return:
        """
        qa_input = {
            'question': question,
            'context': input_text
        }
        res = self.nlp(qa_input)
        output = {
            'question': question,
            'answer': res['answer'],
            'score': res["score"]
        }
        return output

    def get_status_from_text(self, input_text):
        """
        Get status from input text using QA model.
        :param input_text:
        :return:
        """
        answer = self.get_answer('what is the status report?', input_text)
        return answer['answer']

    @staticmethod
    def get_match_from_list(input_text, input_list):
        match = process.extract(input_text, input_list)
        return match[0][0]

    @staticmethod
    def get_vacation_type(input_text):
        types = ['paid', 'loss of pay', 'compensatory']
        match = process.extract(input_text, types)
        if match[0][1] > 60:
            return match[0][0]
        else:
            return 'paid'

    def get_date_from_text(self, text, num_days):
        try:
            if 'forenoon' in text:
                text = text.replace('forenoon', 'morning')
            todays_date = date.today()

            holiday_list = []
            real_time = self.sutime.parse(text)
            print(real_time)
            if len(real_time) == 2:
                print(1)
                start = real_time[0]['value']
                end = real_time[1]['value']
            else:

                answer = real_time[0]['value']
                if len(answer) == 2:
                    start = answer['begin']
                    end = answer['end']
                    print(2)
                else:
                    start = end = answer
                    print(3)
            # print(start, end)
            forenoon_startDay = afternoon_startDay = afternoon_endDay = forenoon_endDay = False

            if 'TAF' in start:
                afternoon_startDay = True

            if 'TMO' in end:
                forenoon_endDay = True

            new_start_date = start.split('T')[0]
            new_end_date = end.split('T')[0]

            if 'XXXX' in new_start_date:
                new_start_date = new_start_date.replace('XXXX', str(todays_date.year))
            if 'XXXX' in new_end_date:
                new_end_date = new_end_date.replace('XXXX', str(todays_date.year))

            # Number of days
            numDays = num_days
            if num_days == 0:
                start_date = datetime.datetime.strptime(new_start_date, "%Y-%m-%d")
                end_date = datetime.datetime.strptime(new_end_date, "%Y-%m-%d")

                delta = end_date - start_date
                days = []
                for i in range(delta.days + 1):
                    day = start_date + timedelta(days=i)
                    strDay = str(day)
                    days.append(strDay.split(' ')[0])

                org_day_list = list(set(days) - set(holiday_list))

                numDays = 0

                for i in org_day_list:
                    strDays = datetime.datetime.strptime(i, "%Y-%m-%d").weekday()
                    if strDays < 5:
                        numDays += 1
                    else:
                        pass

                # same day or diff
                if new_start_date == new_end_date:
                    if afternoon_startDay or forenoon_endDay:
                        numDays -= 0.5
                    else:
                        pass
                else:
                    if afternoon_startDay and forenoon_endDay:
                        numDays -= 1
                    elif afternoon_startDay or forenoon_endDay:
                        numDays -= 0.5
                    else:
                        pass
            else:
                if num_days > 0:
                    if new_start_date == new_end_date:
                        current_date_temp = datetime.datetime.strptime(new_start_date, "%Y-%m-%d")
                        new_end_date = current_date_temp + datetime.timedelta(days=num_days - 1)
                        new_end_date = str(new_end_date)
                        new_end_date = new_end_date.split(' ')[0]

            date_data = {
                "START": {
                    "DATE": new_start_date,
                    "FORENOON": forenoon_startDay,
                    "AFTERNOON": afternoon_startDay
                },
                "END": {
                    "DATE": new_end_date,
                    "FORENOON": forenoon_endDay,
                    "AFTERNOON": afternoon_endDay
                },
                "DAYS": numDays
            }

            return date_data

        except Exception as e:
            Logger.get_logger().exception(e)
            date_data = {
                "START": {
                    "DATE": str(datetime.datetime.today().date()),
                    "FORENOON": False,
                    "AFTERNOON": False
                },
                "END": {
                    "DATE": str(datetime.datetime.today().date()),
                    "FORENOON": False,
                    "AFTERNOON": False
                },
                "DAYS": 1
            }
            return date_data

    @staticmethod
    def get_days_from_text(text):
        texts = text.split(' ')
        for t in texts:
            try:
                days = w2n.word_to_num(t)
                return days
            except ValueError:
                if 'half' in t.lower():
                    return .5
                continue
        return 0

    @staticmethod
    def validate_date_range(data):
        start = data.get('START')['DATE']
        end = data.get('END')['DATE']
        s_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
        e_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()
        if e_date < s_date:
            s_date, e_date = e_date, s_date

        delta = e_date - s_date
        # If delta greater than days replace end with start plus delta
        # if delta > datetime.timedelta(days=float(data.get('DAYS'))):
        #     e_date = s_date + delta - datetime.timedelta(days=1)

        # If delta greater than days replace end with start plus days
        if delta > datetime.timedelta(days=float(data.get('DAYS'))):
            e_date = s_date + datetime.timedelta(days=float(data.get('DAYS'))) - datetime.timedelta(days=1)
        data['START']['DATE'] = s_date
        data['END']['DATE'] = e_date
        return data

    def process_vacation_request(self, input_text, managers=[]):
        # TODO : Validation
        questions = {
            "DAYS": "Number of days?",
            "REASON": "What is the reason for leave?",
            "APPROVER": "Who should receive the request?",
            "START": "Starting date?",
            "TYPE": "Vacation type?",
        }
        data = {}
        num_days = 0
        for key, ques in questions.items():
            try:
                ans = self.get_answer(ques, input_text)['answer']
                if key == 'DAYS':
                    ans = self.get_days_from_text(ans)
                    num_days = ans
                if key == 'APPROVER':
                    if len(managers) > 0:
                        ans = self.get_match_from_list(ans, managers)
                    else:
                        return None
                if key == 'START':
                    ans = self.get_date_from_text(ans, num_days)
                    data.update(ans)
                    continue

                if key == 'TYPE':
                    ans = self.get_vacation_type(ans)

                data[key] = ans
            except Exception as e:
                pass
        return data

    def status_or_vacation(self, input_text):
        """
        Classify input text as vacation request or status report.
        :param input_text:
        :return:
        """
        status_input = {
            'question': "What is the work done?",
            'context': input_text
        }
        leave_input = {
            'question': "When are you on vacation?",
            'context': input_text
        }
        status_score = self.nlp(status_input)["score"]
        leave_score = self.nlp(leave_input)["score"]
        if status_score >= leave_score:
            return {'class': "Status Report", 'confidence': status_score}
        if status_score < leave_score:
            return {'class': "Vacation Request", 'confidence': leave_score}
