import os
import json
import copy
import csv


VLSP_NER_TAGS = ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC']

VLSP_NER_TAG_TO_ID = {
    'O': 0,
    'B-LOC': 1, 
    'I-LOC': 2, 
    'B-ORG': 3, 
    'I-ORG': 4, 
    'B-PER': 5,
    'I-PER': 6, 
    'B-MISC': 7, 
    'I-MISC': 8
}

VLSP_POS_TAGS = ['N', 'Np', 'Nc', 'Nu', 'Ni', 'V', 'A', 'P', 'L', 'M', 'R', 'E', 'C', 'Cc', 'I', 'T',
            'B', 'Ab', 'Cb', 'Eb', 'Mb', 'Nb', 'Pb', 'Vb', 'Y', 'Ny', 'Vy', 'Xy', 'X', 'Z', 'CH']
VLSP_POS_TAG_TO_ID = {
    'N': 0,     # Danh từ
    'Np': 1,    # Danh từ riêng
    'Nc': 2,    # Danh từ chỉ loại
    'Nu': 3,    # Danh từ đơn vị
    'Ni': 4,    # Danh từ ký hiệu
    'V': 5,     # Động từ
    'A': 6,     # Tính từ
    'P': 7,     # Đại từ
    'L': 8,     # Định từ
    'M': 9,     # Số từ
    'R': 10,    # Phó từ
    'E': 11,    # Giới từ
    'C': 12,    # Liên từ
    'Cc': 13,   # Liên từ đẳng lập
    'I': 14,    # Thán từ
    'T': 15,    # Trợ từ, tình thái từ
    'B': 16,    # Từ vay mượn
    'Ab': 17,   # Tính từ vay mượn
    'Cb': 18,   # Liên từ vay mượn
    'Eb': 19,   # Giới từ vay mượn
    'Mb': 20,   # Số từ vay mượn
    'Nb': 21,   # Danh từ vay mượn
    'Pb': 22,   # Đại từ vay mượn
    'Vb': 23,   # Động từ vay mượn
    'Y': 24,    # Từ viết tắt
    'Ny': 25,   # Danh từ viết tắt
    'Vy': 26,   # Động từ viết tắt
    'Xy': 27,   # Từ viết tắt không phân loại được
    'X': 28,    # Từ không phân loại đươc
    'Z': 29,    # Yếu tố cấu tạo từ
    'CH': 30    # Dấu câu
}

def read_json(json_file_path, encoding='utf8'):
    assert json_file_path is not None, 'File path not provided!'
    with open(json_file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data

def save_jsonl(data, save_dir, file_name):
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, "w", encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    pass

def read_tsv(tsv_file_path, encoding='utf8'):
    assert tsv_file_path is not None, 'File path not provided!'
    with open(tsv_file_path, 'r', encoding=encoding) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        data = [line for line in tsv_file]
    return data

class InputExample(object):
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"