DIR_NAME = '../../project/data/'
import os

def CountRows(dirpath):
    total = 0
    for filename_str in os.listdir(dirpath):
        if not filename_str.endswith('.csv'):
            print 'skipping non-csv file: ' + filename_str
            continue
        # filename = open(filename_str)
        count = len(open(DIR_NAME + filename_str).readlines())
        name = os.path.basename(filename_str)
        print(name, count, 'total=', total)
        total += count
    return total

if __name__ == '__main__':
    print CountRows(DIR_NAME)
