import argparse
import os

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str,
        help='data foler to search and delete npy file')
    args = parser.parse_args()
    return args

def main():
    args = parse_argv()
    for (dirpath, dirnames, filenames) in os.walk(args.data_folder):
        for f in filenames:
            if f.endswith('.npy'):
                print(dirpath + '/' + f)
                os.system('rm ' + dirpath + '/' + f)


if __name__ == '__main__':
    main()