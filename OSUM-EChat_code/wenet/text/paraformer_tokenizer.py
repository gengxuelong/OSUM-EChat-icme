def read_seg_dict(path):
    seg_table = {}
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            assert len(arr) == 2
            seg_table[arr[0]] = arr[1]
    return seg_table

