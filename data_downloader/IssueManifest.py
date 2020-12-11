from typing import List, Tuple

import requests

from JSONRecords import Record, Image


class IssueManifest:
    REQUEST_URL = 'https://www.digitalcommonwealth.org/search/'
    MAX_RETRY = 5

    def __init__(self):
        pass

    def _get_manifest(self, issue_id: str) -> requests.request:
        '''
        Get a single issue manifest based on the issue id
        :param issue_id:
        :type issue_id:
        :return:
        :rtype:
        '''
        retry_count = 0
        request_url = self.REQUEST_URL + f'{issue_id}/manifest'
        response = requests.get(request_url)
        while response.status_code != 200 and retry_count < self.MAX_RETRY:
            response = requests.get(request_url)
            retry_count += 1
        if response.status_code != 200:
            raise requests.HTTPError(str(response.status_code) + ' url: ' + response.url)
        return response

    def get_all_manifests(self, issues: List[Record]) -> List[Image]:
        '''
        Finds all issue manifests for the given list of issue ids
        :param issues: List of issues
        :type issues:
        :return:
        :rtype:
        '''
        for issue in issues:
            response = self._get_manifest(issue.issue_id)
            json_data = response.json()
            if json_data.get('sequences') is None:
                raise ValueError('Could not find sequences object on HTTP response')
            sequence = json_data.get('sequences', [])[0]
            canvases = sequence.get('canvases')
            if canvases is None:
                raise ValueError('Could not find canvases object in HTTP response')
            for canvas in canvases:
                image_canvas_url = canvas.get('images')[0].get('resource').get('service').get('@id')
                width = int(canvas.get('images')[0].get('resource').get('width'))
                height = int(canvas.get('images')[0].get('resource').get('height'))
                url_parts = image_canvas_url.split('/')
                image_id = url_parts[-1]
                yield Image(issue, image_id, height, width)
