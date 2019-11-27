import ftplib


def getFile(filename):

    path = 'pubmed/baseline'

    ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov")
    ftp.login("", "")
    ftp.cwd(path)

    try:
        ftp.retrbinary("RETR " + filename[7:], open(filename, 'wb').write)
    except:
        pass

    ftp.quit()


