from typing import List, Optional

import requests


class ImageUrl:
    MAX_RETRY_COUNT = 5
    BASE_IMAGE_URL = 'https://iiif.digitalcommonwealth.org/iiif/2/{image_id}/full/max/0/default.jpg'

    def __init__(self):
        pass

    def build_image_url(self, image_id: str) -> str:
        return self.BASE_IMAGE_URL.format(image_id=image_id)

    def get_verify_image_url(self, image_id: str) -> Optional[str]:
        retry_count = 0
        image_url = self.build_image_url(image_id)
        response_head = requests.head(image_url)
        while response_head.status_code != 200 and retry_count < self.MAX_RETRY_COUNT:
            response_head = requests.head(image_url)
            retry_count += 1
        if response_head.status_code != 200:
            return None
        return image_url

    def get_all_image_urls(self, image_manifest_ids: List[str]) -> List[str]:
        for image_manifest_id in image_manifest_ids:
            yield self.build_image_url(image_manifest_id)
