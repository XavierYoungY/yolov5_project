import yaml

with open('data/Objects365.yaml','r') as stream:
    data=yaml.safe_load(stream)
    names_list=[data['names'][key] for key in data['names']]
    out=''
    for a in names_list:
        out += '\'' + a + '\''+','
    out=out.rstrip(',')+']'
    print(out)

    with open('data/365.yaml','w') as f:
        yaml.dump(out, f)
