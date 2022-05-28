from settings import HOST

URLS = {
    'CLASSIFY-TEXT-NER': '/classify-text-ner',
    'CLASSIFY-TEXT-QA': '/classify-text-qa',
    'GET-EMPLOYEE': '/clean-employee-details',
    'STATUS-NER': '/status-report-ner',
    'STATUS-QA': '/status-from-text-qa',
    'CORRECT-SENTENCE': '/correct-sentence',
    'VACATION-QA': '/vacation-request-qa',
    'PROCESS': '/process'
}

# urls = list(map(lambda x: (x[0], HOST + x[1]), URLS.items()))
# URLS = dict(urls)
