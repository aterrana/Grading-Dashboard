'''
Return the HTTP headers needed to make requests to the specified domain.

Call one of the following:

 - get_http_headers('forum.minerva.edu')
 - get_http_headers('coursebuilder.minerva.edu')

Note: This will make a test call to the specified domain to check that the
credentials work. If they don't an error is raised.
'''

import re
import requests

# The "insert_cookie_cred" will get automatically be replaced by the cookie credentials
# provided as an argument for get_assignment_scores_and_comments.py
domain_curl = {

    'forum.minerva.edu': '''curl 'https://forum.minerva.edu/api/v1/users/self' \
  -H 'accept: application/json, text/javascript, */*; q=0.01' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'cookie: insert_cookie_cred' \
  -H 'dnt: 1' \
  -H 'priority: u=0, i' \
  -H 'referer: https://forum.minerva.edu/app/' \
  -H 'sec-ch-ua: "Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-origin' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36' \
  -H 'x-csrftoken: REDACTED' \
  -H 'x-requested-with: XMLHttpRequest'
  ''',

    'coursebuilder.minerva.edu': '''curl 'https://coursebuilder.minerva.edu/api/v2/core/dashboard/data/?term_name=Spring%202024' \
  -H 'authority: coursebuilder.minerva.edu' \
  -H 'accept: */*' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'cookie: REDACTED' \
  -H 'dnt: 1' \
  -H 'referer: https://coursebuilder.minerva.edu/' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-origin' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
  ''',

}

domain_api_end_point = {
    'forum.minerva.edu': 'https://forum.minerva.edu/api/v1/users/self',
    'coursebuilder.minerva.edu': 'https://coursebuilder.minerva.edu/api/v2/core/dashboard/my_orgs/',
}


def get_http_headers(domain, cookie_cred):
    this_domain_curl = domain_curl[domain].replace("insert_cookie_cred", cookie_cred)
    http_headers = dict(
        (match.group(1), match.group(2))
        for match in re.finditer(r"-H '([^:]*): ([^']*)'", this_domain_curl))

    response = requests.get(domain_api_end_point[domain], headers=http_headers)
    data = response.json()
    if data.get('detail') == 'Authentication credentials were not provided.':
        raise ValueError('Could not authenticate. Check credentials.')

    return http_headers
