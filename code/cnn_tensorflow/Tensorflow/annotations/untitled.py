import pandas as pd
filename = 'test_labels.csv'

file = pd.read_csv(filename,header=None)
categories = file[5].unique()
end = '\n'
s = ' '
class_map = {}
for ID, name in enumerate(categories):
    out = ''
    out += 'item' + s + '{' + end
    out += s*2 + 'id:' + ' ' + (str(ID+1)) + end
    out += s*2 + 'name:' + ' ' + '\'' + name + '\'' + end
    out += '}' + end*2
    

    with open(output_name, 'a') as f:
        f.write(out)
        
    class_map[name] = ID+1

    # text_file.write(out)
    # text_file.close()

