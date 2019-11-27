import glob
import gzip
import xml.etree.ElementTree as eT


def parse(filename):

    document, tag = [], []

    files = glob.glob(filename)

    for file in files:
        pub = gzip.open(file, 'rb')
        try:

            tree = eT.parse(pub)
            root = tree.getroot()
            # print(root)

            for i, elem in enumerate(root):
                if True:

                    for subelem in elem.findall("MedlineCitation"):

                        flag = 0
                        docline = ''

                        # el = subelem.find("PMID").text
                        # print('\n ID: ', el)

                        if subelem.find("Article/ArticleTitle") is None:
                            docline += '#'

                        else:
                            docline += subelem.find("Article/ArticleTitle").text

                        if subelem.find("Article/Abstract/AbstractText") is None:
                            docline += '#'
                            flag = 1

                        else:
                            docline += ' ' + subelem.find("Article/Abstract/AbstractText").text

                        tagline = ''
                        for meselem in subelem.findall("MeshHeadingList/MeshHeading"):
                            try:
                                descrn = meselem.find('DescriptorName').attrib['UI']
                                if meselem.find('DescriptorName').attrib['MajorTopicYN'] == 'Y':

                                    # print('1: ', descrn)
                                    if tagline:
                                        tagline += ', ' + descrn
                                    else:
                                        tagline += descrn
                                else:

                                    for el in meselem.findall('QualifierName'):

                                        if el.attrib['MajorTopicYN'] == 'Y':
                                            # print('2 ', descrn)
                                            if tagline:
                                                tagline += ', ' + descrn
                                            else:
                                                tagline += descrn

                            except:
                                pass

                        if flag == 0:
                            document.append(docline)
                            tag.append(tagline)

        except:
            pass

        for i, item in enumerate(tag):
            if not item:
                tag.pop(i)
                document.pop(i)

    return document, tag
