options = {
    'dataset':[],
    'model':[],
    'model size':[],
    'epochs':[],
    'training rate':[],
    'mbatch size':[],
    'do aug':[],
    'nb aug':[]
    }
state = {
    'dataset':'',
    'model':'',
    'model size':'',
    'epochs':'',
    'training rate':'',
    'mbatch size':'',
    'do aug':'',
    'nb aug':''
}
with open('model_tests/test_params.txt', 'r') as optfile:
    for line in optfile:
        d = line.index(':')
        opt = line[:d]
        vals = line[d+1:-1].split(',')
        options[opt] = vals
param = sys.argv[1]
with open('model_tests/defaults.txt', 'r') as default_state:
    i = 0
    for line in default_state:
        opt = list(options.keys())[i]
        val = int(line)
        if opt != param:
            state[opt] = options[opt][val]
        i += 1
for i in range(len(options[param])):
    state[param] = options[param][i]

    print('test', i)
    print('state:\n',state)
