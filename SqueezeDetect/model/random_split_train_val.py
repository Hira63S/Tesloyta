import os
import numpy as np


def main():
    """
    Divides the idx into train and val files
    """
    image_set_dir = 'kitti_data/image_sets'
    trainval_file = os.path.join(image_set_dir, 'trainval.txt')
    train_file = os.path.join(image_set_dir, 'train.txt')
    val_file = os.path.join(image_set_dir, 'val.txt')

    idx = []
    with open(trainval_file) as f:
        for line in f:
            idx.append(line.strip())

    idx = np.random.permutation(idx)

    third_idx = round(len(idx)*2/3)

    train_idx = sorted(idx[:third_idx])
    val_idx = sorted(idx[third_idx:])

    with open(train_file, 'w') as f:
        for i in train_idx:
            f.write('{}\n'.format(i))

    with open(val_file, 'w') as f:
        for i in val_idx:
            f.write('{}\n'.format(i))

    print('Training set is saved to {}. Length of file is: {} labels.'.format(train_file, len(train_idx)))
    print('Training set is saved to {}. Length of file is: {} labels.'.format(val_file, len(val_idx)))

if __name__ == '__main__':
    np.random.seed(42)
    main()
