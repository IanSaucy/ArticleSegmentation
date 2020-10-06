from typing import List, Tuple

import requests


class IssueManifest:
    REQUEST_URL = 'https://www.digitalcommonwealth.org/search/'

    def __init__(self):
        pass

    def _get_manifest(self, issue_id: str) -> requests.request:
        request_url = self.REQUEST_URL + f'{issue_id}/manifest'
        response = requests.get(request_url)
        if response.status_code != 200:
            raise requests.HTTPError(str(response.status_code) + ' url: ' + response.url)
        return response

    def get_all_manifests(self, issue_ids: List[str]) -> List[Tuple[str, str]]:
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
