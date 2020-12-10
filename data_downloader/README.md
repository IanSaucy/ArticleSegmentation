# Data Downloader
This module helps with scraping the data from the online commonwealth archives. 

It is made up of two parts, first the scraping of the image URLs, which are saved to a csv. 
Then a second step of actually downloading the images specified in the CSV. 

## Setup
Everything is in a `Pipefile`. You need python `>=3.8` and pipenv installed. 
In the folder run the following:
- `pipenv update`
- `pipenv shell`
- `python desired_module.py`

## Scraping URLs

This part is in the aptly named `scrape_urls_driver.py` file. Just call it via python
and it will do the rest. It overwrites the output file each time and thus has to re-scrape
all the URLs on each run.

## Scraping Images

This part can be run via `python image_downloader_driver.py`. It expects the CSV file
from the previous step to be in the same relative path as itself. In addition, the folder:
`./output_images/` needs to exist. It will **not** re-download images that are already in that folder.
Thus, it can pickup where it left off and only download new/missed images. 

As a note, this API has to generate the requested images on the fly, thus it can take a bit of time
and should ideally be run outside of normal business hours. Lastly, you might see some images
go much faster than others, this is simply because they've been cached by the API.