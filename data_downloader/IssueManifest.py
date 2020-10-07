from typing import List, Tuple

import requests


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

    def get_all_manifests(self, issue_ids: List[str]) -> List[Tuple[str, str]]:
        '''
        Finds all issue manifests for the given list of issue ids
        :param issue_ids:
        :type issue_ids:
        :return:
        :rtype:
        '''
        for issue in issue_ids:
            response = self._get_manifest(issue)
            json_data = response.json()
            if json_data.get('sequences') is None:
                raise ValueError('Could not find sequences object on HTTP response')
            sequence = json_data.get('sequences', [])[0]
            canvases = sequence.get('canvases')
            if canvases is None:
                raise ValueError('Could not find canvases object in HTTP response')
            for canvas in canvases:
                image_canvas_url = canvas.get('images')[0].get('resource').get('service').get('@id')
                url_parts = image_canvas_url.split('/')
                yield issue, url_parts[-1]
