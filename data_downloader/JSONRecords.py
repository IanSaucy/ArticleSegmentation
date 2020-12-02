import requests
import json


class JSONRecords:
    MAX_RETRY_COUNT = 5
    def __init__(self, url: str):
        self._url = url

    def _get_record(self, **kwargs) -> requests.request:
        '''
        Obtains a single issue/record from the api
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        '''
        retry_count = 0
        response = requests.get(self._url, params=kwargs)
        while response.status_code != 200 and retry_count < self.MAX_RETRY_COUNT:
            response = requests.get(self._url, params=kwargs)
            retry_count += 1
        return response

    def get_all_records(self):
        '''
        Gets all records from api. Returns a list of issue ids
        :return: issue ids
        :rtype:
        '''
        current_page = 1
        has_next_page = True
        while has_next_page:
            result = self._get_record(page=current_page)
            # Verify that the response was successful
            if result.status_code != 200:
                raise requests.exceptions.HTTPError
            json_data = result.json()
            # We verify that we actually got data back
            if json_data.get('response') is None:
                raise ValueError('Could not find response object in in HTTP response')
            if json_data.get('response').get('docs') is None:
                raise ValueError('Could not find response.docs object in HTTP response')
            if json_data.get('response').get('pages') is None:
                raise ValueError('Could not find response.pages object in HTTP response')
            if json_data.get('response').get('pages').get('last_page?') is None:
                raise ValueError('Could not find last_page object in HTTP response')
            # Process each individual doc ID
            for doc in json_data.get('response').get('docs', []):
                yield doc['id']
            # Results are paginated from the API, we need to iterate over them until
            # the last page indicator
            if not json_data.get('response').get('pages').get('last_page?'):
                current_page = json_data.get('response').get('pages').get('next_page')
            else:
                has_next_page = False
