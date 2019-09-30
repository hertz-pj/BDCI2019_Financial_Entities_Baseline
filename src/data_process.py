import pandas as pd
import re
import os

output_dir = "../process_data/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def clean_str(input):

    input = input.replace(",", "，")
    input = input.replace("\xa0", "")
    input = input.replace("\b", "")
    input = input.replace('"', "")
    input = re.sub("\t|\n|\x0b|\x1c|\x1d|\x1e", "", input)
    input = input.strip()
    input = re.sub('\?\?+','',input)
    input = re.sub('\{IMG:.?.?.?\}','',input)
    input = re.sub('\t|\n','', input)
    
    return input

def main():
    train_df = pd.read_csv("../raw_data/Train_Data.csv", encoding="utf-8-sig")
    test_df = pd.read_csv("../raw_data/Test_Data.csv", encoding="utf-8-sig")

    train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
    test_df['text'] =  test_df['title'].fillna('') + test_df['text'].fillna('')

    train_df['text'] = train_df['text'].apply(clean_str)
    test_df['text'] = test_df['text'].apply(clean_str)

    train_df = train_df[~train_df['unknownEntities'].isnull()]

    # 所有的非中文英文数字符号
    additional_chars = set()
    for t in list(test_df.text) + list(train_df.text):
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t))
    
    # 一些需要保留的符号
    extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
    additional_chars = additional_chars.difference(extra_chars)

    def remove_additional_chars(input):
        for x in additional_chars:
            input = input.replace(x, "")
        return input

    train_df["text"] = train_df["text"].apply(remove_additional_chars)
    test_df["text"] = test_df["text"].apply(remove_additional_chars)

    with open(f"{output_dir}/train.txt", "w", encoding="utf-8") as f:
        for row in train_df.itertuples():
            
            text_lbl = row.text
            entitys = str(row.unknownEntities).split(';')
            for entity in entitys:
                text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-1)*'Ж')

            for c1, c2 in zip(row.text, text_lbl):
                if c2 == 'Ё':
                    f.write('{0}\t{1}\n'.format(c1, 'B'))
                elif c2 == 'Ж':
                    f.write('{0}\t{1}\n'.format(c1, 'I'))
                else:
                    f.write('{0}\t{1}\n'.format(c1, 'O'))
            
            f.write('\n')

    with open(f"{output_dir}/dev.txt", "w", encoding="utf-8") as f:
        for row in train_df.iloc[-200:].itertuples():
            
            text_lbl = row.text
            entitys = str(row.unknownEntities).split(';')
            for entity in entitys:
                text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-1)*'Ж')

            for c1, c2 in zip(row.text, text_lbl):
                if c2 == 'Ё':
                    f.write('{0}\t{1}\n'.format(c1, 'B'))
                elif c2 == 'Ж':
                    f.write('{0}\t{1}\n'.format(c1, 'I'))
                else:
                    f.write('{0}\t{1}\n'.format(c1, 'O'))
            
            f.write('\n')

    with open(f"{output_dir}/test.txt", "w", encoding="utf-8") as f:
        for row in test_df.itertuples():
            text_lbl = row.text
            for c1 in text_lbl:
                f.write('{0}\t{1}\n'.format(c1, 'O'))
            
            f.write('\n')

if __name__ == "__main__":
    main()
    pass