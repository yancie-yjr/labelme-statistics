import os, sys
import json


def updata_json(file, value,to_value):
    key = "label"
    json_file = open(file, encoding='gbk')
    json_data = json.load(json_file)
    i=0
    while i < len(json_data["shapes"]):
        if json_data["shapes"][i]["label"] == value:
            json_data["shapes"][i]["label"]=to_value
        i+=1
    json_file.close()
    return json_data


def write_json(file, json_data):
    with open(file, 'w') as f1:
        json.dump(json_data, f1)
    f1.close()


if __name__ == '__main__':
    print("======输入字母y更改文件夹下所有json文本label的值，输入文件名则更改单一文件的label值！")
    command=input("请输入操作对象：")
    v=input("请输入需要修改的原始值：")
    to_v=input("请输入需要修改的最终值：")
    if command=="y":
        for root, dirs, files in os.walk(os.getcwd()):
            for f in files:
                if ".json" in f:
                    data=updata_json(f,v,to_v)
                    write_json(f,data)
    else:
        data = updata_json(command,v,to_v)
        write_json(command, data)
