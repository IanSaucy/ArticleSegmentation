import csv

from ImageUrl import ImageUrl
from IssueManifest import IssueManifest
from JSONRecords import JSONRecords

if __name__ == '__main__':
    url = 'https://www.digitalcommonwealth.org/search.json?f%5Bcollection_name_ssim%5D%5B%5D=The+Liberator+%28Boston%2C' \
          '+Mass.+%3A+1831-1865%29&f%5Binstitution_name_ssim%5D%5B%5D=Boston+Public+Library&per_page=100 '

    records_tool = JSONRecords(url)
    manifest_tool = IssueManifest()
    image_url_tool = ImageUrl()
    all_manifests = manifest_tool.get_all_manifests(records_tool.get_all_records())
    csv_fields = ['issue_id', 'image_id', 'image_url']
    with open('output_urls.csv', 'w', newline='') as f:
        total = 0
        issue_counter_set = set()
        csvWriter = csv.writer(f)
        csvWriter.writerow(csv_fields)
        for issue_id, image_id in all_manifests:
            issue_counter_set.add(issue_id)
            image_url = image_url_tool.get_verify_image_url(image_id)
            csvWriter.writerow((issue_id, image_id, image_url))
            total += 1
            print(f'Total scraped: {total}', end='\r')
        csvWriter.writerow((len(issue_counter_set), total))
