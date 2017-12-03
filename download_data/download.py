# url = 'https://www.quandl.com/api/v3/datasets/{stock_code}?&api_key={api_key}'
url = 'https://www.quandl.com/api/v3/datasets/{stock_code}.csv?api_key={api_key}'
# stock_code = 'BSE/BOM517421'
with open('../../api_key') as fp:
    api_key = fp.readlines()[0].strip()
formatted_url = url.format(stock_code=stock_code, api_key=api_key)
# print formatted_url

codes = []
with open('BSE-datasets-codes.csv') as fp:
    for line in fp:
        codes.append(line.split(',')[0])
# print (codes[0], "len=", len(codes))

def download_code(code):
    file_name = 'data/' + code.replace('/', '_') + '.csv'
    if os.path.isfile(file_name):
        print '{} already exists'.format(code)
        return
    print 'downloading ' + str(code)
    formatted_url = url.format(stock_code=code, api_key=api_key)
    resp = urllib2.urlopen(formatted_url)

    with open(file_name, 'w') as fp:
        fp.write(resp.read())

from time import sleep

btime = 1
def backoff():
    global btime
    if btime <= 120:
        btime += 1
    return btime

def reset_backoff():
    print 'restting backoff to 1 sec'
    global btime
    btime = 1


with open('errors.log', 'r') as err_file:
    lines = err_file.readlines()

new_errors = open('new_errors.log', 'w')

print 'no of errors: %d' % len(lines)
count = 1
for line in lines:
    code = line.split('|')[0].strip()

    while(True):
        try:
            download_code(code)
            reset_backoff()
            break
        except Exception as e:
            print e
            if isinstance(e, urllib2.HTTPError):
                if e.code == 429:
                    btime = backoff()
                    print 'freakin slow site. Sleeping for %d seconds' % btime
                    sleep(btime)
                    continue
                else:
                    print 'wtf' + str(e)
                    new_errors.write(code + ' | ' + str(e) + '\n')
                    break
            else:
                print 'really wtf' + str(e)
                new_errors.write(code + ' | ' + str(e) + '\n')
                break
    # end while
    print 'Downloaded file {x} of {y}'.format(x=count, y=len(lines))
    count += 1

new_errors.close()
