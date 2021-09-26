import csv
import copy
import math
import re


class ID3Solver:
    most_label_value = ''
    gain_method = ''
    max_depth = ''

    def __init__(self, label, gain_method, max_depth):
        self.most_label_value = label
        self.gain_method = gain_method
        self.max_depth = max_depth
        
    # calculate the information gain
    def get_gain(self, examples, attr, values):
        # get the proportions of different label values
        def get_label_prop_dict(sub_examples):
            res = {}
            for example in sub_examples:
                if example['label'] in res.keys():
                    res[example['label']] += 1
                else:
                    res[example['label']] = 1
            for key in res:
                res[key] /= len(sub_examples)
            return res

        def Entropy(sub_examples):
            res = 0
            label_prop_dict = get_label_prop_dict(sub_examples)
            for key in label_prop_dict:
                val = label_prop_dict[key]
                res -= val * math.log(val, 2)
            return res

        def ME(sub_examples):
            if len(sub_examples) == 0:
                return 0
            else:
                label_prop_dict = get_label_prop_dict(sub_examples)
                vals = label_prop_dict.values()
                return 1-max(vals)
            
        def GI(sub_examples):
            res = 1
            label_prop_dict = get_label_prop_dict(sub_examples)
            for key in label_prop_dict:
                val = label_prop_dict[key]
                res -= val*val
            return res
        
        def get_purity(examples):
            if self.gain_method == 0:
                return Entropy(examples)
            elif self.gain_method == 1:
                return ME(examples)
            elif self.gain_method == 2:
                return GI(examples)
            else:
                return Entropy(examples)

        # purity: Examples
        purity_examples = get_purity(examples)

        # purity: subExamples
        purity_values = []
        proportions = []
        num_Exmaples = len(examples)
        for value in values:
            sub_examples = []
            for example in examples:
                if example[attr] == value:
                    sub_examples.append(example)               
            purity_values.append(get_purity(sub_examples))
            proportions.append(len(sub_examples)/num_Exmaples)
        expected_purity = 0
        for purity_value, proportion in zip(purity_values, proportions):
            expected_purity += purity_value*proportion

        gain = purity_examples - expected_purity
        return gain

    # find the best attributes to split the examples
    def get_best_attr(self, examples, attrs):
        gains = {}
        # calculate information gains for all attrs
        for attr, values in attrs.items():
            gains[attr] = self.get_gain(examples, attr, values)
        # select the best one
        best_attr = ''
        best_attr_gain = -10
        for attr, gain in gains.items():
            if gain > best_attr_gain:
                best_attr_gain = gain
                best_attr = attr
        return best_attr

    # get examples that satisfy: attr = attr_v
    def get_sub_examples(self, examples, attr, attr_v):
        sub_examples = []
        for example in examples:
            if example[attr] == attr_v:
                sub_examples.append(example)
        return sub_examples

    # test if the sub_example has the same label
    def has_same_label(self, examples):
        label = examples[0]['label']
        for example in examples:
            if example['label'] != label:
                return False
        return True

    # get the most common feature
    def get_most_common_label(self, examples):
        label_values = ['unacc', 'acc', 'good', 'vgood']
        # get the most common label
        label_num = [0, 0, 0, 0]
        for example in examples:
            lb = example['label']
            if lb == 'unacc':
                label_num[0] += 1
            elif lb == 'acc':
                label_num[1] += 1
            elif lb == 'good':
                label_num[2] += 1
            elif lb == 'vgood':
                label_num[3] += 1
        most_label_value = label_values[label_num.index(max(label_num))]

        return most_label_value

    # run ID3, genearte the decision tree
    def ID3(self, examples, attrs, node):
        # check it is the max_depth, make it a leaf
        if node['depth'] == self.max_depth:
            node['is_leaf'] = True
            node['label'] = self.get_most_common_label(examples)
            return

        # find the best attr to split the examples, then add a root node to this tree
        attr = self.get_best_attr(examples, attrs)           # the selected attr

        sub_attrs = copy.deepcopy(attrs)
        del sub_attrs[attr]
        # add node to the structure
        node['feature'] = attr
        node['values'] = {}
        node['is_leaf'] = False
        
        # for each value of the attr, get a sub_example, and generate a branch for it
        for attr_v in attrs[attr]:
            # add a branch
            node['values'][attr_v] = {}
            cur_node = node['values'][attr_v]
            cur_node['depth'] = node['depth']+1

            sub_examples = self.get_sub_examples(examples, attr, attr_v)

            # if the subset is empty, then make the subset as a leaf, and set the most common label
            if len(sub_examples) == 0:
                # add a leaf node
                cur_node['is_leaf'] = True
                cur_node['label'] = self.get_most_common_label(examples)
                
            # else if the subset has the same label, make the subset as a leaf and set the common label
            elif self.has_same_label(sub_examples):
                cur_node['is_leaf'] = True
                cur_node['label'] = sub_examples[0]['label']

            # else if the subset has different labels, then run ID3() recursively  
            else:
                self.ID3(sub_examples, sub_attrs, cur_node)


class Training:
    gain_method = ''
    max_depth = ''
    examples = ''

    def __init__(self, gain_method, max_depth, examples):
        self.gain_method = gain_method
        self.max_depth = max_depth
        self.examples = examples

    def train(self):
        attrs = {'buying': ['vhigh', 'high', 'med', 'low'], 'maint': ['vhigh', 'high', 'med', 'low'], 
            'doors': ['2', '3', '4', '5more'], 'persons': ['2', '4', 'more'], 'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high']}

        label_values = ['unacc', 'acc', 'good', 'vgood']

        # get the most common label
        label_num = [0, 0, 0, 0]
        for example in self.examples:
            lb = example['label']
            if lb == 'unacc':
                label_num[0] += 1
            elif lb == 'acc':
                label_num[1] += 1
            elif lb == 'good':
                label_num[2] += 1
            elif lb == 'vgood':
                label_num[3] += 1

        most_label_value = label_values[label_num.index(max(label_num))]

        decision_tree = {'depth': 0}

        ID3_obj = ID3Solver(most_label_value, self.gain_method, self.max_depth)
        ID3_obj.ID3(self.examples, attrs, decision_tree)

        return decision_tree

'''
generate the training example
example = [{'attr1': , 'attr2': , 'attr3': }, ...],
'''
def process_data(file_name):
    examples = []
    with open(file_name, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            example = {}
            example['buying'] = row[0]
            example['maint'] = row[1]
            example['doors'] = row[2]
            example['persons'] = row[3]
            example['lug_boot'] = row[4]
            example['safety'] = row[5]
            example['label'] = row[6]
            examples.append(example)
    return examples

# test the data
def test_data(tree, test_examples):
    num = len(test_examples)
    correct_num = 0

    for example in test_examples:
        cur_node = tree
        while not cur_node['is_leaf']:
            cur_node = cur_node['values'][example[cur_node['feature']]]
        predict_label = cur_node['label']
        real_label = example['label']
        if predict_label == real_label:
            correct_num += 1
    
    return 1 - correct_num/num

if __name__ == '__main__':
    '''
    gain method: 0-entropy, 1-ME, 2-GI
    '''
    train_examples = process_data('./data/car/train.csv')
    test_examples = process_data('./data/car/test.csv')

    print('Information Gain')
    print('--------------------------------------------')
    print('MaxDepth\tTrainingError\tTestError')
    for i in range(6):
        training_obj = Training(0, i+1, train_examples)
        tree = training_obj.train()
        train_error = test_data(tree, train_examples)
        test_error = test_data(tree, test_examples)
        print((i+1), '\t', train_error, '\t', test_error)
    
    print()

    print('Majority Error')
    print('--------------------------------------------')
    print('MaxDepth\tTrainingError\tTestError')
    for i in range(6):
        training_obj = Training(1, i+1, train_examples)
        tree = training_obj.train()
        train_error = test_data(tree, train_examples)
        test_error = test_data(tree, test_examples)
        print((i+1), '\t',train_error, '\t',test_error)

    print()

    print('Gini Index')
    print('--------------------------------------------')
    print('MaxDepth\tTrainingError\tTestError')
    for i in range(6):
        training_obj = Training(2, i+1, train_examples)
        tree = training_obj.train()
        train_error = test_data(tree, train_examples)
        test_error = test_data(tree, test_examples)
        print((i+1), '\t', train_error, '\t', test_error)
