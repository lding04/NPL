from npl import crawl


def test_crawl():
    args = crawl.parser.parse_args(['-f', './test/data/template.csv'])
    crawler = crawl.Crawler(args)
    crawler.crawl()
