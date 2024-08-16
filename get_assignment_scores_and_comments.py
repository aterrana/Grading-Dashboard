'''
Retrieve the scores and written comments for all assignment submissions in all
sections of a course given a Forum assignment id.

You can get the Forum id of an assignment by visiting a section page on Forum
and clicking on the title of an assignment. This will take you to a URL of the
form https://forum.minerva.edu/app/assignments/123456 where 123456 is the id.
'''
import io
import sys
import json
import datetime
import requests
from requests_futures.sessions import FuturesSession
from collections import defaultdict
import random
import copy
import dashboard

import credentials

# Provide cookie credentials
try:
    cookie_cred = sys.argv[1]
except:
    cookie_cred = None
if cookie_cred is None:
    sys.stderr.write(f'''
Usage: {sys.argv[0]} cookie_cred assignment_id

Must include a cookie credential string
''')
    sys.exit(-1)

# Provide the assignment id as an input argument
try:
    assignment_id = int(sys.argv[2])
except:
    assignment_id = None
if assignment_id is None:
    sys.stderr.write(f'''
Usage: {sys.argv[0]} cookie_cred assignment_id

Retrieve the scores and written comments for all assignment submissions in all sections of a course given a Forum assignment id.

The assignment_id argument must be an integer.
''')
    sys.exit(-1)


http_headers = credentials.get_http_headers('forum.minerva.edu', cookie_cred)
requests_session = FuturesSession()


class RequestException(Exception):
    def __init__(self, response, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_response = response


def get_futures_with_retry(urls, http_headers, retries=2):
    # Asynchronously retrieve a list of URLs.
    global requests_session
    futures = [requests_session.get(url, headers=http_headers) for url in urls]
    outputs = []
    retry_index = []
    for future in futures:
        response = future.result()
        if response.status_code != 200:
            if 500 <= response.status_code < 600:
                # Server error. Retry in case it's temporary.
                retry_index.append(len(outputs))
            outputs.append(RequestException(response))
        else:
            outputs.append(response.json())

    # Retry failed GETs
    if retries > 0:
        retry_urls = [outputs[i].request_response.request.url for i in retry_index]
        retry_outputs = get_futures_with_retry(retry_urls, http_headers, retries - 1)
        for i, output in zip(retry_index, retry_outputs):
            outputs[i] = output
    elif len(retry_index) > 0:
        print('ERROR: Failed GETs:', file=sys.stderr)
        for i in retry_index:
            print(' -', outputs[i].request_response.request.url, '->', outputs[i].request_response.status_code, file=sys.stderr)

    return outputs


# Step 1: Get section id for the given assignment id
print('Looking up course for assignment id', assignment_id)
response = requests.get(
    f'https://forum.minerva.edu/api/v1/assignments/{assignment_id}/nested_for_detail_page',
    headers=http_headers)
assignment = response.json()
section_id = assignment['section-id']

# In Forum, the same assignment will have different Forum ids in different
# sections of the same course. This Course Builder id is unique for the same
# assignment across all sections of a course.
course_builder_id = assignment['num']

# Step 2: Get course id for the section id
response = requests.get(
    f'https://forum.minerva.edu/api/v1/sections/{section_id}',
    headers=http_headers)
section = response.json()
course_id = section['course-id']

# Step 3: Get learning outcomes for the course
response = requests.get(
    f'https://forum.minerva.edu/api/v1/courses/{course_id}/trees',
    headers=http_headers)
hc_and_lo_trees = response.json()
los = {
    x['id']: x['hashtag']
    for los in hc_and_lo_trees['lo-tree']['course-objectives']
    for x in los['learning-outcomes']}

# Step 4: Get all sections for the course id
response = requests.get(
    f'https://forum.minerva.edu/api/v1/sections?course-id={course_id}&all-possible=true&registrar-dashboard=true&state=all&hide-defaults=true',
    headers=http_headers)
sections = response.json()
course_code = sections[0]['course']['course-code']
sections = {
    s['id']: {'title': s['title'], 'student_count': s['student-count']}
    for s in sections}
print(f'Found course {course_id}. {course_code} with section ids {list(sections.keys())}')

# Step 5: Get all assignments for all sections of the course
all_assignments = dict(zip(sections.keys(), get_futures_with_retry(
    [
        f'https://forum.minerva.edu/api/v1/paginated-assignments?filter_section_id={section_id}&category=iterative&category=signature&category=final&category=location-based&include_description=false&grader_list=true&show_unmuted=true&title_matches=&page_size=100'
        for section_id in sections.keys()],
    http_headers)))

# In each section, find the Forum assignment id matching `course_builder_id`
for section_id in sections.keys():
    result = all_assignments[section_id]
    all_assignments[section_id] = None
    for assignment in result['results']:
        if assignment['num'] == course_builder_id:
            all_assignments[section_id] = assignment['id']

# Step 6: Fetch the grading data for the assignment in each section
all_assignments = dict(zip(
    sections.keys(),
    get_futures_with_retry(
        [
            f'https://forum.minerva.edu/api/v1/assignments/{all_assignments[section_id]}/nested_for_grader'
            for section_id in sections.keys()],
        http_headers)))

grades = defaultdict(lambda: defaultdict(list))
grades = {}
for section_id in sections.keys():
    assignment = all_assignments[section_id]
    # Get scores and comments, grouped by student
    for assessment in assignment['outcome-assessments']:
        if assessment['active']:  # As far as I know, deleted scores/comments are marked as inactive
            student_id = assessment['target-user-id']
            
            if section_id not in grades.keys():
                grades[section_id] = {}
            if student_id not in grades[section_id].keys():
                grades[section_id][student_id] = []

            grades[section_id][student_id].append({
                'learning_outcome': los.get(assessment['learning-outcome'], assessment['learning-outcome']),
                'score': assessment['score'],
                'comment': assessment['comment'],
                'graded_blindly': assessment['graded-blindly'],
                'created_on': assessment['created-on'],
                'updated_on': assessment['updated-on']})

output = {
    'course': {
        'id': course_id,
        'code': course_code},
    'sections': sections,  # {section_id: {'title': TTh name (11am), 'student_count': ...}}
    'grades': grades}  # {section_id: {student_id: grade_data}}

with open("grade_data.py", 'w', encoding="utf-8") as file:
    file.write(str(output))
print("Grade data collected")

dashboard.create_report()

'''
# Step 7 (optional): Anonymize the data
fake_section_ids = defaultdict(lambda: random.randint(10000000, 20000000))
fake_student_ids = defaultdict(lambda: random.randint(10000000, 20000000))
fake_data = defaultdict(dict)
for section_id, section_grades in grades.items():
    for student_id, student_grades in section_grades.items():
        fake_data[fake_section_ids[section_id]][fake_student_ids[student_id]] = copy.deepcopy(student_grades)
        for grade in fake_data[fake_section_ids[section_id]][fake_student_ids[student_id]]:
            if len(grade['comment']) >= 5:
                grade['comment'] = fake.text(max_nb_chars=len(grade['comment']))

# Step 8: Write the data to a file
with open(f'fake_data_{assignment_id}', 'wt') as fp:
    fp.write(str(dict(fake_data)))
    fp.write('\n')
'''
