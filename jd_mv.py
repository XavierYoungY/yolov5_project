import shutil


if __name__=='__main__':
    img_path = 'data/datasets/jd/images/val/'
    name_list = 'data/datasets/jd/labels/val.txt'
    with open(name_list, 'r') as f:
        name_list = f.readlines()
        for path in name_list:
            path = path.strip()
            name = path.split('/')[-1]

            new_path = img_path + name
            shutil.copy(path.replace('/val',''), new_path)
    print('VAL_')

    img_path = 'data/datasets/jd/images/train/'
    name_list = 'data/datasets/jd/labels/train.txt'
    with open(name_list,'r') as f:
        name_list = f.readlines()
        for path in name_list:
            path=path.strip()
            name = path.split('/')[-1]

            new_path = img_path + name
            shutil.copy(path.replace('/train', ''), new_path)
    print('train')
