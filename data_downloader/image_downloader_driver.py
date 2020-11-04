from source_images import download_all_images

if __name__ == '__main__':
    success, failures = download_all_images('./output_urls.csv', './output_images/', './output_errors.csv')
    print(f'Image download finished with {success} successes and {failures} failures')