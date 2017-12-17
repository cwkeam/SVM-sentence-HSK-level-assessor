def clean(arr):
    results = []
    for line in arr:
        new_str = line.replace('，', '')
        new_str = new_str.replace('：', '')
        new_str = new_str.replace('！', '')
        new_str = new_str.replace(' ', '')
        new_str = new_str.replace('。', '')
        new_str = new_str.replace('？', '')
        new_str = new_str.replace('）', '')
        new_str = new_str.replace('（', '')
        new_str = new_str.replace('”', '')
        new_str = new_str.replace('“', '')
        new_str = new_str.replace('\n', '')
        new_str = new_str + ","

        new_str = ''.join([i for i in new_str if not i.isdigit()])
        results.append(new_str)
    return results
