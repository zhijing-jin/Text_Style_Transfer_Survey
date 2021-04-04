class Constants:
    data_dir = './'
    csv_style_transfer = data_dir + 'paper_list_style_transfer.csv'
    md_style_transfer = data_dir + 'paper_list_style_transfer.md'
    md_format = '1. ({year} {venue}) **{title}.** _{authors}_. [[paper]({paper})]'
    elem_video = '[[video]({})]'
    elem_code = '[[code]({})]'
    elem_code_unofficial = '[[unofficial code]({})]'
    elem_data = '[[data - {}]({})]'
    elem_note = '{}'


class CSV2MDConverter:
    def __init__(self):
        self.papers = []

    def load_papers(self, csv_file):
        from efficiency.log import read_csv
        papers = read_csv(csv_file)
        self.papers = [row for row in papers if row['title']]

    def count_authors(self):
        authors = [i['authors'].split(', ') for i in self.papers]
        from efficiency.function import flatten_list
        authors = flatten_list(authors)
        from collections import Counter
        cnt = Counter(authors)
        from efficiency.log import show_var
        show_var(['cnt'])
        import pdb;
        pdb.set_trace()

    def csv2category_n_formatted_markdown(self, csv_file):
        self.load_papers(csv_file)

        from collections import defaultdict
        category2md_elems = defaultdict(list)
        for csv_line in self.papers:
            category2md_elems[csv_line['category']].append(self._csv_line2md_elem(csv_line))

        self.save_to_md(category2md_elems)

    def _csv_line2md_elem(self, csv_line):
        venue = csv_line['venue']
        if csv_line['top_note']: venue += '; ' + csv_line['top_note']

        md = C.md_format.format(year=csv_line['year'], venue=venue,
                                title=csv_line['title'], authors=csv_line['authors'],
                                paper=csv_line['paper'], )
        if csv_line['video']: md += ' ' + C.elem_video.format(csv_line['video'])
        if csv_line['code']: md += ' ' + C.elem_code.format(csv_line['code'])
        if csv_line['unofficial_code']: md += ' ' + C.elem_code_unofficial.format(csv_line['unofficial_code'])
        if csv_line['data']:
            dataset_name, dataset_link = csv_line['data'].split('http')
            md += ' ' + C.elem_data.format(dataset_name, 'http' + dataset_link)
        if csv_line['note']: md += ' ' + C.elem_note.format(csv_line['note'])
        return md

    def save_to_md(self, category2md_elems):
        category2md = {k: '\n'.join(v) for k, v in category2md_elems.items()}
        output = []
        for k, v in category2md.items():
            output.append('## ' + k)
            output.append(v)
        from efficiency.log import fwrite
        fwrite('\n'.join(output), C.md_style_transfer)


def main():
    try:
        import efficiency
    except:
        import os
        os.system('pip install efficiency')

    csv2md_converter = CSV2MDConverter()
    csv2md_converter.csv2category_n_formatted_markdown(C.csv_style_transfer)

    # csv2md_converter.count_authors()


if __name__ == '__main__':
    C = Constants()
    main()
