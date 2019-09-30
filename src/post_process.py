import pandas as pd 
import re

old_entities = []
train_df = pd.read_csv("../raw_data/Train_Data.csv", encoding="utf-8-sig")
for x in list(train_df["unknownEntities"].fillna("")):
    old_entities.extend(x.split(";"))
old_entities = set(old_entities)
add_char = {']', '：', '~', '！', '%', '[', '《', '】', ';', '”', ':', '》', '？', '>', '/', '#', '。', '；', '&', '=', '，', '“', '【'}

def islegitimate(x):
    if re.findall("\\"+"|\\".join(add_char), x):
        return False
    if x in old_entities:
        return False

    return True


def extract_entity(res):
    entity_list = []
    entity = ""
    sent = []
    
    cnt = 0
    for r in res:
        if r == "":
            if len(entity) > 1:
                for e in sent:
                    if e.find(entity) == 0:
                        entity = ""
                        break
                if entity != "":
                    sent.append(entity)
                entity = ""
            cnt += 1
            entity_list.append(sent)
            sent = []
        elif r[4] == "B":
            if entity != "":
                if len(entity) > 1:
                    sent.append(entity)
                entity = ""
            entity += r[0]
        elif r[4] == "I" and entity != "":
            entity += r[0]
        elif r[4] == "O":
            if entity != "" and len(entity) > 1:
                sent.append(entity)
            entity = ""
    return entity_list

def main():

    with open("../model/label_test_final_repro.txt", "r", encoding="utf-8") as f:
        res = [line.strip() for line in f.readlines()]

    entity_list = extract_entity(res)

    new_entities = []

    for entities in entity_list:
        new = []
        for e in entities:
            if islegitimate(e):
                new.append(e)
        new_entities.append(new)

    Dt = pd.read_csv("../raw_data/Test_Data.csv", encoding="utf-8-sig")
    assert len(Dt) == len(new_entities)
    Dt["unknownEntities"] = [";".join(set(x)).replace(",","") for x in new_entities]
    Dt[["id", "unknownEntities"]].to_csv(f"../result/submit.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
  
    pass
    