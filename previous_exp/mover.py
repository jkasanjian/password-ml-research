import os
import shutil
import glob



def move_files(s):
    try:
        os.mkdir(s + '/models')
        os.mkdir(s + '/graphs')
    except FileExistsError:
        pass

    files = glob.glob(s + '/**/*.png', recursive=True)
    for f in files:
        os.replace(f, s + '/graphs/' + f.split('/')[-1])

    files = glob.glob(s + '/*.joblib', recursive=True)
    for f in files:
        os.replace(f, s + '/models/' + f.split('/')[-1])
    
    try:
        os.rmdir(s + '/LOG_PNG')
        os.rmdir(s + '/RF_PNG')
        os.rmdir(s + '/KNN_PNG')
        os.rmdir(s + '/SVM_PNG')
    except FileNotFoundError:
        pass

    
    

def main():
    subs = os.listdir()
    subs.remove('mover.py')
    subs.remove('s002')
    subs.remove('s027')
    
    for s in subs:
        if s[0] != '.':
            move_files(s)


main()