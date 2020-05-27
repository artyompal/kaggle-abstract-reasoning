ENABLE_MULTIPROCESSING = True

from dsl import cpp_trace_param_automata

def generate_public_submission():
    import numpy as np
    import pandas as pd

    import os
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    from xgboost import XGBClassifier
    import pdb

    # data_path = Path('.')

    data_path = Path('.')
    if not (data_path / 'test').exists():
        data_path = Path('../input/abstraction-and-reasoning-challenge')
    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'

    def plot_result(test_input, test_prediction,
                    input_shape):
        """
        Plots the first train and test pairs of a specified task,
        using same color scheme as the ARC app
        """
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        test_input = test_input.reshape(input_shape[0], input_shape[1])
        axs[0].imshow(test_input, cmap=cmap, norm=norm)
        axs[0].axis('off')
        axs[0].set_title('Actual Target')
        test_prediction = test_prediction.reshape(input_shape[0], input_shape[1])
        axs[1].imshow(test_prediction, cmap=cmap, norm=norm)
        axs[1].axis('off')
        axs[1].set_title('Model Prediction')
        plt.tight_layout()
        plt.show()


    def plot_test(test_prediction, task_name):
        """
        Plots the first train and test pairs of a specified task,
        using same color scheme as the ARC app
        """
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        fig, axs = plt.subplots(1, 1, figsize=(15, 15))
        axs.imshow(test_prediction, cmap=cmap, norm=norm)
        axs.axis('off')
        axs.set_title(f'Test Prediction {task_name}')
        plt.tight_layout()
        plt.show()


    # https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
    def flattener(pred):
        str_pred = str([row for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred


    sample_sub1 = pd.read_csv(data_path / 'sample_submission.csv')
    sample_sub1 = sample_sub1.set_index('output_id')
    sample_sub1.head()


    def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):
        if cur_row <= 0:
            top = -1
        else:
            top = color[cur_row - 1][cur_col]

        if cur_row >= nrows - 1:
            bottom = -1
        else:
            bottom = color[cur_row + 1][cur_col]

        if cur_col <= 0:
            left = -1
        else:
            left = color[cur_row][cur_col - 1]

        if cur_col >= ncols - 1:
            right = -1
        else:
            right = color[cur_row][cur_col + 1]

        return top, bottom, left, right


    def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
        if cur_row == 0:
            top_left = -1
            top_right = -1
        else:
            if cur_col == 0:
                top_left = -1
            else:
                top_left = color[cur_row - 1][cur_col - 1]
            if cur_col == ncols - 1:
                top_right = -1
            else:
                top_right = color[cur_row - 1][cur_col + 1]

        return top_left, top_right


    def make_features(input_color, nfeat):
        nrows, ncols = input_color.shape
        feat = np.zeros((nrows * ncols, nfeat))
        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0] = i
                feat[cur_idx, 1] = j
                feat[cur_idx, 2] = input_color[i][j]
                feat[cur_idx, 3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11] = (i + j)
                feat[cur_idx, 12] = len(np.unique(input_color[i - local_neighb:i + local_neighb,
                                                  j - local_neighb:j + local_neighb]))

                cur_idx += 1

        return feat


    def features(task, mode='train'):
        num_train_pairs = len(task[mode])
        feat, target = [], []

        global local_neighb
        for task_num in range(num_train_pairs):
            input_color = np.array(task[mode][task_num]['input'])

            target_color = task[mode][task_num]['output']

            nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

            target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])

            if (target_rows != nrows) or (target_cols != ncols):
                print('Number of input rows:', nrows, 'cols:', ncols)
                print('Number of target rows:', target_rows, 'cols:', target_cols)
                not_valid = 1
                return None, None, 1

            imsize = nrows * ncols
            # offset = imsize*task_num*3 #since we are using three types of aug
            feat.extend(make_features(input_color, nfeat))
            target.extend(np.array(target_color).reshape(-1, ))

        return np.array(feat), np.array(target), 0


    # mode = 'eval'
    mode = 'test'
    if mode == 'eval':
        task_path = evaluation_path
    elif mode == 'train':
        task_path = training_path
    elif mode == 'test':
        task_path = test_path

    all_task_ids = sorted(os.listdir(task_path))

    nfeat = 13
    local_neighb = 5
    valid_scores = {}

    model_accuracies = {'ens': []}
    pred_taskids = []

    for task_id in all_task_ids:

        task_file = str(task_path / task_id)
        with open(task_file, 'r') as f:
            task = json.load(f)

        feat, target, not_valid = features(task)
        if not_valid:
            print('ignoring task', task_file)
            print()
            not_valid = 0
            continue

        xgb = XGBClassifier(n_estimators=10, n_jobs=-1)
        xgb.fit(feat, target, verbose=-1)

        #     training on input pairs is done.
        #     test predictions begins here

        num_test_pairs = len(task['test'])
        for task_num in range(num_test_pairs):
            cur_idx = 0
            input_color = np.array(task['test'][task_num]['input'])
            nrows, ncols = len(task['test'][task_num]['input']), len(
                task['test'][task_num]['input'][0])
            feat = make_features(input_color, nfeat)

            print('Made predictions for ', task_id[:-5])

            preds = xgb.predict(feat).reshape(nrows, ncols)

            if (mode == 'train') or (mode == 'eval'):
                ens_acc = (np.array(task['test'][task_num]['output']) == preds).sum() / (nrows * ncols)

                model_accuracies['ens'].append(ens_acc)

                pred_taskids.append(f'{task_id[:-5]}_{task_num}')

            #             print('ensemble accuracy',(np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols))
            #             print()

            preds = preds.astype(int).tolist()
            #         plot_test(preds, task_id)
            sample_sub1.loc[f'{task_id[:-5]}_{task_num}',
                            'output'] = flattener(preds)

    if (mode == 'train') or (mode == 'eval'):
        df = pd.DataFrame(model_accuracies, index=pred_taskids)
        print(df.head(10))

        print(df.describe())
        for c in df.columns:
            print(f'for {c} no. of complete tasks is', (df.loc[:, c] == 1).sum())

        df.to_csv('ens_acc.csv')

    sample_sub1.head()

    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'
    training_tasks = sorted(os.listdir(training_path))
    eval_tasks = sorted(os.listdir(evaluation_path))

    T = training_tasks
    Trains = []
    for i in range(400):
        task_file = str(training_path / T[i])
        task = json.load(open(task_file, 'r'))
        Trains.append(task)

    E = eval_tasks
    Evals = []
    for i in range(400):
        task_file = str(evaluation_path / E[i])
        task = json.load(open(task_file, 'r'))
        Evals.append(task)

    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    # 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
    # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    plt.figure(figsize=(5, 2), dpi=200)
    plt.imshow([list(range(10))], cmap=cmap, norm=norm)
    plt.xticks(list(range(10)))
    plt.yticks([])
    # plt.show()

    def plot_task(task):
        n = len(task["train"]) + len(task["test"])
        fig, axs = plt.subplots(2, n, figsize=(4 * n, 8), dpi=50)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig_num = 0
        for i, t in enumerate(task["train"]):
            t_in, t_out = np.array(t["input"]), np.array(t["output"])
            axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
            axs[0][fig_num].set_title(f'Train-{i} in')
            axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
            axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
            axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
            axs[1][fig_num].set_title(f'Train-{i} out')
            axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
            axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
            fig_num += 1
        for i, t in enumerate(task["test"]):
            t_in, t_out = np.array(t["input"]), np.array(t["output"])
            axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
            axs[0][fig_num].set_title(f'Test-{i} in')
            axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
            axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
            axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
            axs[1][fig_num].set_title(f'Test-{i} out')
            axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
            axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
            fig_num += 1

        plt.tight_layout()
        plt.show()

    def plot_picture(x):
        plt.imshow(np.array(x), cmap=cmap, norm=norm)
        plt.show()

    def Defensive_Copy(A):
        n = len(A)
        k = len(A[0])
        L = np.zeros((n, k), dtype=int)
        for i in range(n):
            for j in range(k):
                L[i, j] = 0 + A[i][j]
        return L.tolist()

    def Create(task, task_id=0):
        n = len(task['train'])
        Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
        Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
        Input.append(Defensive_Copy(task['test'][task_id]['input']))
        return Input, Output

    def Recolor(task):
        Input = task[0]
        Output = task[1]
        Test_Picture = Input[-1]
        Input = Input[:-1]
        N = len(Input)

        for x, y in zip(Input, Output):
            if len(x) != len(y) or len(x[0]) != len(y[0]):
                return -1

        Best_Dict = -1
        Best_Q1 = -1
        Best_Q2 = -1
        Best_v = -1
        # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
        Pairs = []
        for t in range(15):
            for Q1 in range(1, 8):
                for Q2 in range(1, 8):
                    if Q1 + Q2 == t:
                        Pairs.append((Q1, Q2))

        for Q1, Q2 in Pairs:
            for v in range(4):

                if Best_Dict != -1:
                    continue
                possible = True
                Dict = {}

                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v == 2:
                                p1 = i % Q1
                            else:
                                p1 = (n - 1 - i) % Q1
                            if v == 0 or v == 3:
                                p2 = j % Q2
                            else:
                                p2 = (k - 1 - j) % Q2
                            color1 = x[i][j]
                            color2 = y[i][j]
                            if color1 != color2:
                                rule = (p1, p2, color1)
                                if rule not in Dict:
                                    Dict[rule] = color2
                                elif Dict[rule] != color2:
                                    possible = False
                if possible:

                    # Let's see if we actually solve the problem
                    for x, y in zip(Input, Output):
                        n = len(x)
                        k = len(x[0])
                        for i in range(n):
                            for j in range(k):
                                if v == 0 or v == 2:
                                    p1 = i % Q1
                                else:
                                    p1 = (n - 1 - i) % Q1
                                if v == 0 or v == 3:
                                    p2 = j % Q2
                                else:
                                    p2 = (k - 1 - j) % Q2

                                color1 = x[i][j]
                                rule = (p1, p2, color1)

                                if rule in Dict:
                                    color2 = 0 + Dict[rule]
                                else:
                                    color2 = 0 + y[i][j]
                                if color2 != y[i][j]:
                                    possible = False
                    if possible:
                        Best_Dict = Dict
                        Best_Q1 = Q1
                        Best_Q2 = Q2
                        Best_v = v

        if Best_Dict == -1:
            return -1  # meaning that we didn't find a rule that works for the traning cases

        # Otherwise there is a rule: so let's use it:
        n = len(Test_Picture)
        k = len(Test_Picture[0])

        answer = np.zeros((n, k), dtype=int)

        for i in range(n):
            for j in range(k):
                if Best_v == 0 or Best_v == 2:
                    p1 = i % Best_Q1
                else:
                    p1 = (n - 1 - i) % Best_Q1
                if Best_v == 0 or Best_v == 3:
                    p2 = j % Best_Q2
                else:
                    p2 = (k - 1 - j) % Best_Q2

                color1 = Test_Picture[i][j]
                rule = (p1, p2, color1)
                if (p1, p2, color1) in Best_Dict:
                    answer[i][j] = 0 + Best_Dict[rule]
                else:
                    answer[i][j] = 0 + color1

        return answer.tolist()

    sample_sub2 = pd.read_csv(data_path / 'sample_submission.csv')
    sample_sub2.head()

    def flattener(pred):
        str_pred = str([row for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred

    example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # display(example_grid)
    print(flattener(example_grid))

    Solved = []
    Problems = sample_sub2['output_id'].values
    Proposed_Answers = []

    test_paths_my = {task.stem: json.load(task.open()) for task in test_path.iterdir()}
    test_task_ids = np.sort(list(test_paths_my.keys()))
    print(Problems, len(Problems))
    task_number_my = dict(zip(test_task_ids, np.arange(100)))

    for i in range(len(Problems)):
        output_id = Problems[i]
        task_id = output_id.split('_')[0]

        pair_id = int(output_id.split('_')[1])
        f = str(test_path / str(task_id + '.json'))

        with open(f, 'r') as read_file:
            task = json.load(read_file)

        n = len(task['train'])
        Input = [Defensive_Copy(task['train'][j]['input']) for j in range(n)]
        Output = [Defensive_Copy(task['train'][j]['output']) for j in range(n)]
        Input.append(Defensive_Copy(task['test'][pair_id]['input']))

        solution = Recolor([Input, Output])

        pred = ''

        if solution != -1:
            Solved.append(i)
            pred1 = flattener(solution)
            pred = pred + pred1 + ' '

        if pred == '':
            pred = flattener(example_grid)

        Proposed_Answers.append(pred)

    sample_sub2['output'] = Proposed_Answers

    sample_sub1 = sample_sub1.reset_index()
    sample_sub1 = sample_sub1.sort_values(by="output_id")

    sample_sub2 = sample_sub2.sort_values(by="output_id")
    out1 = sample_sub1["output"].astype(str).values
    out2 = sample_sub2["output"].astype(str).values

    merge_output = []
    for o1, o2 in zip(out1, out2):
        o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:2]
        o = " ".join(o[:3])
        merge_output.append(o)
    sample_sub1["output"] = merge_output
    sample_sub1["output"] = sample_sub1["output"].astype(str)

    # test_paths_my = { task.stem: json.load(task.open()) for task in test_path.iterdir() }
    # test_task_ids = np.sort(list(test_paths_my.keys()))
    # task_number_my = dict(zip(test_task_ids, np.arange(100)))
    submission = sample_sub1.copy()
    submission.to_csv("public_submission.csv", index=False)


#generate_public_submission()

import numpy as np

from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
import time
from collections import defaultdict
import os
import json
import random
import copy
import networkx as nx
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt

from itertools import product
import pandas as pd
import multiprocessing
import subprocess

# from moviepy.editor import ImageSequenceClip
# from moviepy.editor import clips_array, CompositeVideoClip
# from moviepy.video.io.html_tools import html_embed, HTML2

# def display_vid(vid, verbose=False, **html_kw):
#     """
#         Display a moviepy video clip, useful for removing loadbars
#     """

#     rd_kwargs = {
#         'fps': 10, 'verbose': verbose
#     }

#     if not verbose:
#         rd_kwargs['logger'] = None

#     return HTML2(html_embed(vid, filetype=None, maxduration=60,
#                             center=True, rd_kwargs=rd_kwargs, **html_kw))

data_path = Path('../input/abstraction-and-reasoning-challenge/')
# data_path = Path('.') # Artyom: it's better use symlinks locally
cmap_lookup = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

cmap_lookup = [np.array([int(x[1:3], 16), int(x[3:5], 16), int(x[5:], 16)]) for x in cmap_lookup]

def cmap(x):
    """
        Translate a task matrix to a color coded version

        arguments
            x : a h x w task matrix
        returns
            a h x w x 3 matrix with colors instead of numbers
    """
    y = np.zeros((*x.shape, 3))
    y[x < 0, :] = np.array([112, 128, 144])
    y[x > 9, :] = np.array([255, 248, 220])
    for i, c in enumerate(cmap_lookup):
        y[x == i, :] = c
    return y

def draw_one(x, k=20):
    """
        Create a PIL image from a task matrix, the task will be
        drawn using the default color coding with grid lines

        arguments
            x : a task matrix
            k = 20 : an up scaling factor
        returns
            a PIL image

    """
    img = Image.fromarray(cmap(x).astype(np.uint8)).resize((x.shape[1] * k, x.shape[0] * k), Image.NEAREST)

    draw = ImageDraw.Draw(img)
    for i in range(x.shape[0]):
        draw.line((0, i * k, img.width, i * k), fill=(80, 80, 80), width=1)
    for j in range(x.shape[1]):
        draw.line((j * k, 0, j * k, img.height), fill=(80, 80, 80), width=1)
    return img

def vcat_imgs(imgs, border=10):
    """
        Concatenate images vertically

        arguments:
            imgs : an array of PIL images
            border = 10 : the size of space between images
        returns:
            a PIL image
    """

    h = max(img.height for img in imgs)
    w = sum(img.width for img in imgs)
    res_img = Image.new('RGB', (w + border * (len(imgs) - 1), h), color=(255, 255, 255))

    offset = 0
    for img in imgs:
        res_img.paste(img, (offset, 0))
        offset += img.width + border

    return res_img

def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(n * 4, 8))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0

    def go(ax, title, x):
        ax.imshow(draw_one(x), interpolation='nearest')
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])

    for i, t in enumerate(task["train"]):
        go(axs[0][fig_num], f'Train-{i} in', t["input"])
        go(axs[1][fig_num], f'Train-{i} out', t["output"])
        fig_num += 1
    for i, t in enumerate(task["test"]):
        go(axs[0][fig_num], f'Test-{i} in', t["input"])
        try:
            go(axs[1][fig_num], f'Test-{i} out', t["output"])
        except:
            go(axs[1][fig_num], f'Test-{i} out', np.zeros_like(t["input"]))
        fig_num += 1

    plt.tight_layout()
    plt.show()

def real_trace_param_automata(input, params, n_iter, n_hidden):
    """
        Execute an automata and return all the intermediate states

        arguments:
            step_fn : transition rule function, should take two arguments `input` and `hidden_i`,
                should return an output grid an a new hidden hidden grid
            n_iter : num of iteration to perform
            n_hidden: number of hidden grids, if set to 0 `hidden_i` will be set to None
            laodbar = True: weather display loadbars
        returns:
            an array of tuples if output and hidden grids
    """

    # hidden = np.zeros((n_hidden, *input.shape)) if n_hidden > 0 else None
    #
    # global_rules, ca_rules = params
    #
    # trace = [(input, hidden)]
    #
    # for rule in global_rules:
    #
    #     output, hidden = apply_rule(input, hidden, rule)
    #     trace.append((output, hidden))
    #     input = output
    #
    # its = range(n_iter)
    #
    # for i_it in its:
    #     output, hidden = compute_parametrized_automata(input, hidden, ca_rules)
    #     trace.append((output, hidden))
    #
    #     if (input.shape == output.shape) and (output == input).all():
    #         break
    #     input = output

    hidden = np.zeros((n_hidden, *input.shape)) if n_hidden > 0 else None

    global_rules, ca_rules, split_rule, merge_rule = params

    grids = apply_split_rule(input, hidden, split_rule)
    #print(grids[0][0])

    for rule in global_rules:
        for i, (inp, hid) in enumerate(grids):
            if rule['macro_type'] == 'global_rule':
                if rule['apply_to'] == 'all' or \
                    (rule['apply_to'] == 'index' and i == rule['apply_to_index']%len(grids) or
                    (rule['apply_to'] == 'last' and i == len(grids) - 1)):
                    grids[i] = apply_rule(inp, hid, rule)
            elif rule['macro_type'] == 'global_interaction_rule':
                grids = apply_interaction_rule(grids, rule)
        #print(grids[0][0])
    #1/0
    for i, (input, hidden) in enumerate(grids):
        for _ in range(n_iter):
            output, hidden = compute_parametrized_automata(input, hidden, ca_rules)

            if np.array_equal(input, output):
                break

            input = output

        grids[i] = (output, hidden)
    output = apply_merge_rule(grids, merge_rule, split_rule)

    return output

def apply_interaction_rule(grids, rule):
    if rule['type'] == 'align_pattern':
        # index_from = rule['index_from'] % len(grids)
        # index_to = rule['index_to'] % len(grids)
        # allow_rotation = rule['allow_rotation']
        if len(grids) > 5:
            return grids
        for index_from in range(len(grids)):
            for index_to in range(index_from+1, len(grids)):
                input_i = grids[index_from][0]
                input_j = grids[index_to][0]
                # print(np.max(input_i>0, axis=1))
                # print(np.max(input_i>0, axis=1).shape)
                # print(np.arange(input_i.shape[0]).shape)
                #1/0
                i_nonzero_rows = np.arange(input_i.shape[0])[np.max(input_i>0, axis=1)]
                i_nonzero_columns = np.arange(input_i.shape[1])[np.max(input_i>0, axis=0)]
                j_nonzero_rows = np.arange(input_j.shape[0])[np.max(input_j>0, axis=1)]
                j_nonzero_columns = np.arange(input_j.shape[1])[np.max(input_j>0, axis=0)]
                if i_nonzero_rows.shape[0] == 0 or i_nonzero_columns.shape[0] == 0 or \
                                j_nonzero_rows.shape[0] == 0 or j_nonzero_columns.shape[0] == 0:
                    continue
                i_minrow = np.min(i_nonzero_rows)
                i_mincol = np.min(i_nonzero_columns)
                i_maxrow = np.max(i_nonzero_rows) + 1
                i_maxcol = np.max(i_nonzero_columns) + 1
                j_minrow = np.min(j_nonzero_rows)
                j_mincol = np.min(j_nonzero_columns)
                j_maxrow = np.max(j_nonzero_rows) + 1
                j_maxcol = np.max(j_nonzero_columns) + 1

                figure_to_align = input_i[i_minrow:i_maxrow, i_mincol:i_maxcol]
                figure_target = input_j[j_minrow:j_maxrow, j_mincol:j_maxcol]

                best_fit = 0
                best_i_fit, best_j_fit = -1, -1
                #print(figure_to_align)
                #print(figure_target)
                if figure_to_align.shape[0] < figure_target.shape[0] or figure_to_align.shape[1] < figure_target.shape[1]:
                    continue
                    #1/0
                else:
                    for i_start in range((figure_to_align.shape[0] - figure_target.shape[0])+1):
                        for j_start in range((figure_to_align.shape[1] - figure_target.shape[1])+1):
                            fig_1 = figure_to_align[i_start:(i_start + figure_target.shape[0]), j_start:(j_start + figure_target.shape[1])]
                            if np.logical_and(np.logical_and(figure_target > 0, figure_target!=rule['allow_color']), figure_target != fig_1).any():
                                continue
                            fit = np.sum(figure_target==fig_1)
                            if fit > best_fit:
                                best_i_fit, best_j_fit = i_start, j_start
                                best_fit = fit

                    if best_fit == 0:
                        continue

                    imin = j_minrow-best_i_fit
                    imax = j_minrow-best_i_fit + figure_to_align.shape[0]
                    jmin = j_mincol - best_j_fit
                    jmax = j_mincol - best_j_fit + figure_to_align.shape[1]

                    begin_i = max(imin, 0)
                    begin_j = max(jmin, 0)
                    end_i = min(imax, input_j.shape[0])
                    end_j = min(jmax, input_j.shape[1])

                    i_fig_begin = (begin_i-imin)
                    i_fig_end = figure_to_align.shape[0]-(imax-end_i)
                    j_fig_begin = (begin_j-jmin)
                    j_fig_end = figure_to_align.shape[1]-(jmax-end_j)
                    if rule['fill_with_color'] == 0:
                        input_j[begin_i:end_i, begin_j:end_j] = figure_to_align[i_fig_begin:i_fig_end, j_fig_begin:j_fig_end]
                    else:
                        for i, j in product(range(end_i-begin_i + 1), range(end_j-begin_j + 1)):
                            if input_j[begin_i + i, begin_j + j] == 0:
                               input_j[begin_i + i, begin_j + j] = rule['fill_with_color'] * (figure_to_align[i_fig_begin + i, j_fig_begin + j])


    return grids




def trace_param_automata(input, params, n_iter, n_hidden):
    # expected = real_trace_param_automata(input, params, n_iter, n_hidden)
    #
    # testcase = {'input': input, 'params': params}
    # print(str(testcase).replace('\'', '"').replace('array(', '').replace(')', ''))

    output = cpp_trace_param_automata(input, params, n_iter)

    # if not np.array_equal(expected, output):
    #     print('cpp result is wrong')
    #     print('input:')
    #     print(input)
    #     print('expected:')
    #     print(expected)
    #     print('got:')
    #     print(output)
    #
    #     diff = [[str(g) if e != g else '-' for e, g in zip(exp_row, got_row)]
    #              for exp_row, got_row in zip(expected, output)]
    #     diff_lines = [' '.join(line) for line in diff]
    #     diff_str = '[[' + ']\n ['.join(diff_lines)
    #
    #     print('diff:')
    #     print(diff_str)
    #     print('rules')
    #     print(params)
    #
    #     assert False

    return [[output]]

# def vis_automata_trace(states, loadbar=False, prefix_image=None):
#     """
#         Create a video from an array of automata states
#
#         arguments:
#             states : array of automata steps, returned by `trace_automata()`
#             loadbar = True: weather display loadbars
#             prefix_image = None: image to add to the beginning of each frame
#         returns
#             a moviepy ImageSequenceClip
#     """
#     frames = []
#     if loadbar:
#         states = tqdm(states, desc='Frame')
#     for i, (canvas, hidden) in enumerate(states):
#
#         frame = []
#         if prefix_image is not None:
#             frame.append(prefix_image)
#         frame.append(draw_one(canvas))
#         frames.append(vcat_imgs(frame))
#
#     return ImageSequenceClip(list(map(np.array, frames)), fps=10)

# def vis_automata_paramed_task(tasks, parameters, n_iter, n_hidden, vis_only_ix=None):
#     """
#         Visualize the automata steps during the task solution
#         arguments:
#             tasks : the task to be solved by the automata
#             step_fn : automata transition function as passed to `trace_automata()`
#             n_iter : number of iterations to perform
#             n_hidden : number of hidden girds
#     """
#
#     n_vis = 0
#
#     def go(task, n_vis, test=False):
#
#         if vis_only_ix is not None and vis_only_ix != n_vis:
#             return
#         trace = trace_param_automata(task['input'], parameters, n_iter, n_hidden)
#         if not test:
#             vid = vis_automata_trace(trace, prefix_image=draw_one(task['output']))
#         else:
#             vid = vis_automata_trace(trace, prefix_image=draw_one(np.zeros_like(task['input'])))
#
#         # display(display_vid(vid))
#
#     for task in (tasks['train']):
#         n_vis += 1
#         go(task, n_vis)
#
#     for task in (tasks['test']):
#         n_vis += 1
#         go(task, n_vis, True)


training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

def load_data(p, phase=None):
    """
        Load task data

    """
    if phase in {'training', 'test', 'evaluation'}:
        p = data_path / phase / p

    task = json.loads(Path(p).read_text())
    dict_vals_to_np = lambda x: {k: np.array(v) for k, v in x.items()}
    assert set(task) == {'test', 'train'}
    res = dict(test=[], train=[])
    for t in task['train']:
        assert set(t) == {'input', 'output'}
        res['train'].append(dict_vals_to_np(t))
    for t in task['test']:
        if phase == 'test':
            assert set(t) == {'input'}
        else:
            assert set(t) == {'input', 'output'}
        res['test'].append(dict_vals_to_np(t))

    return res

nbh = lambda x, i, j: {
    (ip, jp) : x[i+ip, j+jp]
        for ip, jp in product([1, -1, 0], repeat=2)
            if 0 <= i+ip < x.shape[0] and 0 <= j+jp < x.shape[1] and (not (ip==0 and jp==0))
}

def get_random_split_rule(all_colors, best_candidates={}, temp=0, config={}, r_type=None):
    rule = {}
    rule['type'] = random.choice(['nothing', 'color_figures', 'figures', 'macro_multiply'])
    if rule['type'] in ['color_figures', 'figures']:
        rule['sort'] = random.choice(['biggest', 'smallest'])

    if rule['type'] == 'macro_multiply':
        rule['k1'] = np.random.randint(config['mink1'], config['maxk1']+1)
        rule['k2'] = np.random.randint(config['mink2'], config['maxk2']+1)
    return rule


def get_random_merge_rule(all_colors, best_candidates={}, temp=0, config={}, r_type=None):
    rule = {}
    rule['type'] = random.choice(['cellwise_or', 'output_first', 'output_last'])
    return rule

def apply_split_rule(input, hidden, split_rule):
    if split_rule['type'] == 'nothing':
        return [(input, hidden)]

    if split_rule['type'] == 'macro_multiply':
        ks = split_rule['k1'] *  split_rule['k2']
        grids = [(np.copy(input), np.copy(hidden)) for _ in range(ks)]
        return grids
    #split_rule['type'] = 'figures'
    dif_c_edge = split_rule['type'] == 'figures'
    communities = get_connectivity_info(input, ignore_black=True, edge_for_difcolors=dif_c_edge)

    if len(communities) > 0:
        if split_rule['sort'] == 'biggest':
            communities = communities[::-1]

        grids = [(np.zeros_like(input), np.zeros_like(hidden)) for _ in range(len(communities))]
        for i in range(len(communities)):
            for point in communities[i]:
                grids[i][0][point] = input[point]
    else:
        grids = [(input, hidden)]

    return grids


def apply_merge_rule(grids, merge_rule, split_rule):

    if split_rule['type'] == 'macro_multiply':
        shape_base = grids[0][0].shape
        shapes = [arr[0].shape for arr in grids]
        if not np.array([shape_base == sh for sh in shapes]).all():
            return np.zeros((1, 1), dtype=np.int)

        ks_1 = split_rule['k1']
        ks_2 = split_rule['k2']
        output = np.zeros((shape_base[0] * ks_1, shape_base[1] * ks_2), dtype=np.int8)
        for k1 in range(ks_1):
            for k2 in range(ks_2):
                output[(k1*shape_base[0]):((k1+1) * shape_base[0]), (k2*shape_base[1]):((k2+1) * shape_base[1])] = grids[k1*ks_2 + k2][0]

        return output

    if merge_rule['type'] == 'cellwise_or':
        output = np.zeros_like(grids[0][0])
        for i in np.arange(len(grids))[::-1]:
            if grids[i][0].shape == output.shape:
                output[grids[i][0]>0] = grids[i][0][grids[i][0]>0]
        return output
    elif merge_rule['type'] == 'output_first':
        output = grids[0][0]
    elif merge_rule['type'] == 'output_last':
        output = grids[-1][0]
    return output



def get_random_ca_rule(all_colors, best_candidates={}, temp=0, config={}, r_type=None):
    types_possible = \
        [
            'copy_color_by_direction',
            'direct_check',
            'indirect_check',
            'nbh_check',
            'corner_check',
            'color_distribution',
        ]

    ca_rules = []
    best_candidates_items = list(best_candidates.items())
    if len(best_candidates_items) > 0:
        for best_score, best_candidates_score in best_candidates_items:
            for best_c in best_candidates_score:
                gl, ca, _, _ = best_c
                ca_rules += [c['type'] for c in ca]
        type_counts = dict(zip(types_possible, np.zeros(len(types_possible))))

        rules, counts = np.unique(ca_rules, return_counts=True)
        for i in range(rules.shape[0]):
            type_counts[rules[i]] += counts[i]
        counts = np.array(list(type_counts.values()))
        if np.sum(counts) > 0:
            counts /= np.sum(counts)
        else:
            counts = np.ones(counts.shape[0]) / counts.shape[0]
        uniform = np.ones(counts.shape[0]) / counts.shape[0]
        probs = temp * counts + (1 - temp) * uniform

    else:
        probs = np.ones(len(types_possible)) / len(types_possible)

    colors = all_colors[1:]

    type_probs = np.ones(len(types_possible)) / len(types_possible)

    if r_type is None:
        random_type = types_possible[np.random.choice(len(types_possible), p=probs)]
    else:
        random_type = r_type

    def get_random_out_color():
        possible_colors = config['possible_colors_out']
        return np.random.choice(possible_colors)

    def get_random_ignore_colors():
        if config['possible_ignore_colors'].shape[0] > 0:
            possible_colors = config['possible_ignore_colors']
            return possible_colors[np.random.randint(2, size=possible_colors.shape[0]) == 1]
        else:
            return []

    def get_random_all_colors():
        return all_colors[np.random.randint(2, size=all_colors.shape[0]) == 1]

    def get_random_colors():
        return get_random_all_colors()

    def get_random_all_color():
        return np.random.choice(all_colors)

    def get_random_color():
        return get_random_all_color()

    rule = {}
    rule['type'] = random_type
    rule['macro_type'] = 'ca_rule'
    rule['ignore_colors'] = list(config['ignore_colors'])

    if np.random.rand() < 0.5 and config['possible_ignore_colors'].shape[0]:
        rule['ignore_colors'] += [random.choice(config['possible_ignore_colors'])]

    if random_type == 'copy_color_by_direction':
        rule['direction'] = random.choice(['everywhere'])
        rule['copy_color'] = [get_random_out_color()]
        rule['look_back_color'] = rule['copy_color'][0]

    elif random_type == 'corner_check':
        if np.random.rand() < 0.5:
            rule['nbh_check_colors'] = [get_random_all_color()]
        else:
            rule['nbh_check_colors'] = list(np.unique([get_random_all_color(), get_random_all_color()]))
        rule['nbh_check_out'] = get_random_out_color()
        rule['ignore_colors'] = list(np.unique(rule['ignore_colors'] + [rule['nbh_check_out']]))

    elif random_type == 'direct_check':
        rule['nbh_check_sum'] = np.random.randint(4)
        if np.random.rand() < 0.5:
            rule['nbh_check_colors'] = [get_random_all_color()]
        else:
            rule['nbh_check_colors'] = list(np.unique([get_random_all_color(), get_random_all_color()]))
        rule['nbh_check_out'] = get_random_out_color()
        rule['ignore_colors'] = list(np.unique(rule['ignore_colors'] + [rule['nbh_check_out']]))

    elif random_type == 'indirect_check':
        rule['nbh_check_sum'] = np.random.randint(4)
        if np.random.rand() < 0.5:
            rule['nbh_check_colors'] = [get_random_all_color()]
        else:
            rule['nbh_check_colors'] = list(np.unique([get_random_all_color(), get_random_all_color()]))
        rule['nbh_check_out'] = get_random_out_color()
        rule['ignore_colors'] = list(np.unique(rule['ignore_colors'] + [rule['nbh_check_out']]))

    elif random_type == 'nbh_check':
        rule['nbh_check_sum'] = np.random.randint(8)
        if np.random.rand() < 0.5:
            rule['nbh_check_colors'] = [get_random_all_color()]
        else:
            rule['nbh_check_colors'] = list(np.unique([get_random_all_color(), get_random_all_color()]))
        rule['nbh_check_out'] = get_random_out_color()
        rule['ignore_colors'] = list(np.unique(rule['ignore_colors'] + [rule['nbh_check_out']]))

    elif random_type == 'color_distribution':
        rule['direction'] = random.choice(
            ['top', 'bottom', 'left', 'right', 'top_left', 'bottom_left', 'top_right', 'bottom_right'])
        rule['check_in_empty'] = np.random.randint(2)
        rule['color_out'] = get_random_out_color()
        if rule['check_in_empty'] == 0:
            rule['color_in'] = rule['color_out']
        else:
            rule['color_in'] = get_random_all_color()

        rule['ignore_colors'] = list(np.unique(rule['ignore_colors'] + [rule['color_out']]))

    return rule


def get_random_global_rule(all_colors, best_candidates={}, temp=0, config={}, r_type=None):
    types_possible = \
        [
            'distribute_colors',
            'unity',
            'color_for_inners',
            'map_color',
            'draw_lines',
            'draw_line_to',
            'gravity',
            'make_holes',
            'distribute_from_border',
            'align_pattern',
            'rotate',
            'flip'
        ]

    if config['allow_make_smaller']:
        types_possible += \
            [
                'crop_empty',
                'crop_figure',
                'split_by_H',
                'split_by_W',
                'reduce'
            ]

    # if config['allow_make_bigger']:
        # types_possible += \
            # [
                # 'macro_multiply_by',
                # 'micro_multiply_by',
                # 'macro_multiply_k',
            # ]
    gl_rules = []
    best_candidates_items = list(best_candidates.items())
    if len(best_candidates_items) > 0:
        for best_score, best_candidates_score in best_candidates_items:
            for best_c in best_candidates_score:
                gl, ca, _, _ = best_c
                gl_rules += [c['type'] for c in gl]
        type_counts = dict(zip(types_possible, np.zeros(len(types_possible))))

        rules, counts = np.unique(gl_rules, return_counts=True)
        for i in range(rules.shape[0]):
            type_counts[rules[i]] += counts[i]
        counts = np.array(list(type_counts.values()))
        if np.sum(counts) > 0:
            counts /= np.sum(counts)
        else:
            counts = np.ones(counts.shape[0]) / counts.shape[0]
        uniform = np.ones(counts.shape[0]) / counts.shape[0]
        probs = temp * counts + (1 - temp) * uniform
    else:
        probs = np.ones(len(types_possible)) / len(types_possible)

    colors = all_colors[1:]

    type_probs = np.ones(len(types_possible)) / len(types_possible)

    if r_type is None:
        random_type = types_possible[np.random.choice(len(types_possible), p=probs)]
    else:
        random_type = r_type

    def get_random_all_colors():
        return all_colors[np.random.randint(2, size=all_colors.shape[0]) == 1]

    def get_random_colors():
        return all_colors[np.random.randint(2, size=all_colors.shape[0]) == 1]

    def get_random_all_color():
        return np.random.choice(all_colors)

    def get_random_color():
        return get_random_all_color()

    def get_random_out_color():
        possible_colors = config['possible_colors_out']
        return np.random.choice(possible_colors)

    rule = {}
    rule['type'] = random_type
    rule['macro_type'] = 'global_rule'
    rule['apply_to'] = random.choice(['all', 'index'])

    if np.random.rand()<0.2:
        rule['apply_to'] = 'last'

    if rule['apply_to'] == 'index':
        rule['apply_to_index'] = np.random.choice(10)

    if random_type == 'macro_multiply_k':
        rule['k'] = (np.random.randint(1, 4), np.random.randint(1, 4))
    elif random_type == 'flip':
        rule['how'] = random.choice(['ver', 'hor'])

    elif random_type == 'rotate':
        rule['rotations_count'] = np.random.randint(1, 4)

    elif random_type == 'micro_multiply_by':
        rule['how_many'] = random.choice([2, 3, 4, 5, 'size'])

    elif random_type == 'macro_multiply_by':
        rule['how_many'] = random.choice(['both', 'hor', 'ver'])
        rule['rotates'] = [np.random.randint(1) for _ in range(4)]
        rule['flips'] = [random.choice(['hor', 'ver', 'horver', 'no']) for _ in range(4)]


    elif random_type == 'distribute_from_border':
        rule['colors'] = list(np.unique([get_random_out_color(), get_random_all_color()]))

    elif random_type == 'draw_lines':
        rule['direction'] = random.choice(['everywhere', 'horizontal', 'vertical', 'horver', 'diagonal'])
        # 'top', 'bottom', 'left', 'right',
        # 'top_left', 'bottom_left', 'top_right', 'bottom_right'])
        rule['not_stop_by_color'] = 0  # get_random_all_color()
        rule['start_by_color'] = get_random_all_color()
        rule['with_color'] = get_random_out_color()

    elif random_type == 'reduce':
        rule['skip_color'] = get_random_all_color()
    elif random_type == 'draw_line_to':
        #rule['direction_type'] = random.choice(['border'])

        rule['direction_color'] = get_random_all_color()

        rule['not_stop_by_color'] = 0
        if np.random.rand() < 0.5:
            rule['not_stop_by_color_and_skip'] = get_random_all_color()
        else:
            rule['not_stop_by_color_and_skip'] = 0

        rule['start_by_color'] = get_random_all_color()
        rule['with_color'] = get_random_out_color()

    elif random_type == 'distribute_colors':
        rule['colors'] = list(np.unique([get_random_out_color(), get_random_all_color()]))
        rule['horizontally'] = np.random.randint(2)
        rule['vertically'] = np.random.randint(2)
        rule['intersect'] = get_random_out_color()

    elif random_type == 'color_for_inners':
        rule['color_out'] = get_random_out_color()

    elif random_type == 'crop_figure':
        rule['mode'] = random.choice(['smallest', 'biggest'])
        rule['dif_c_edge'] = random.choice([True, False])


    elif random_type == 'unity':
        rule['mode'] = random.choice(['diagonal', 'horizontal', 'vertical', 'horver'])
        # rule['inner'] = np.random.choice(2)
        rule['ignore_colors'] = [0]
        if np.random.rand() < 0.5:
            rule['ignore_colors'] += [get_random_all_color()]
        rule['with_color'] = random.choice([get_random_out_color(), 0])

    elif random_type == 'map_color':
        rule['color_in'] = get_random_all_color()
        rule['color_out'] = get_random_out_color()

    elif random_type == 'gravity':
        rule['gravity_type'] = random.choice(['figures', 'cells'])
        rule['steps_limit'] = np.random.choice(2)
        rule['look_at_what_to_move'] = np.random.choice(2)
        if rule['look_at_what_to_move'] == 1:
            rule['color_what'] = get_random_out_color()
        rule['direction_type'] = random.choice(['border', 'color'])
        if rule['direction_type'] == 'border':
            rule['direction_border'] = random.choice(['top', 'bottom', 'left', 'right'])
        else:
            rule['direction_color'] = get_random_color()

    elif random_type == 'split_by_H' or random_type == 'split_by_W':
        rule['merge_rule'] = random.choice(['and', 'equal', 'or', 'xor'])

    elif random_type == 'align_pattern':
        rule['macro_type'] = 'global_interaction_rule'
        # rule['allow_rotation'] = False
        rule['allow_color'] = get_random_all_color()
        rule['fill_with_color'] = 0 #random.choice([0, get_random_all_color()])

    return rule


def get_task_metadata(task):
    colors = []
    shapes_input = [[], []]
    shapes_output = [[], []]
    for part in ['train']:
        for uni_task in task[part]:
            inp = uni_task['input']
            colors += list(np.unique(inp))
            out = uni_task['output']
            colors += list(np.unique(out))

            shapes_input[0].append(inp.shape[0])
            shapes_input[1].append(inp.shape[1])
            shapes_output[0].append(out.shape[0])
            shapes_output[1].append(out.shape[1])

    all_colors = np.unique(colors)

    min_k1 = int(np.floor(np.min(np.array(shapes_output[0])/np.array(shapes_input[0]))))
    min_k2 = int(np.floor(np.min(np.array(shapes_output[1])/np.array(shapes_input[1]))))
    max_k1 = int(np.ceil(np.max(np.array(shapes_output[0])/np.array(shapes_input[0]))))
    max_k2 = int(np.ceil(np.max(np.array(shapes_output[1])/np.array(shapes_input[1]))))

    max_shape = np.max([shapes_input])

    config = {}

    config['mink1'] = max(1, min(min(min_k1, 30//max_shape), 3))
    config['mink2'] = max(1, min(min(min_k2, 30//max_shape), 3))
    config['maxk1'] = max(1, min(min(max_k1, 30//max_shape), 3))
    config['maxk2'] = max(1, min(min(max_k2, 30//max_shape), 3))


    config['allow_make_smaller'] = False
    config['allow_make_bigger'] = False

    for uni_task in task['train']:
        if uni_task['input'].shape[0] > uni_task['output'].shape[0] or \
                uni_task['input'].shape[1] > uni_task['output'].shape[1]:
            config['allow_make_smaller'] = True

        if uni_task['input'].shape[0] < uni_task['output'].shape[0] or \
                uni_task['input'].shape[1] < uni_task['output'].shape[1]:
            config['allow_make_bigger'] = True

    colors_out = []
    changed_colors = []
    inp_colors = []
    for uni_task in task['train']:
        inp = uni_task['input']
        out = uni_task['output']
        for i in range(min(inp.shape[0], out.shape[0])):
            for j in range(min(inp.shape[1], out.shape[1])):
                inp_colors.append(inp[i, j])
                if out[i, j] != inp[i, j]:
                    colors_out.append(out[i, j])
                    changed_colors.append(inp[i, j])

    inp_colors = np.unique(inp_colors)
    changed_colors = np.unique(changed_colors)

    config['ignore_colors'] = [c for c in inp_colors if not c in changed_colors]
    config['possible_ignore_colors'] = np.array([c for c in all_colors if not c in config['ignore_colors']])
    if len(colors_out) == 0:
        colors_out = [0]
    config['possible_colors_out'] = np.unique(colors_out)

    return all_colors, config


def compute_parametrized_automata(input, hidden_i, rules):
    output = np.zeros_like(input, dtype=int)

    hidden_o = np.copy(hidden_i)

    for i, j in product(range(input.shape[0]), range(input.shape[1])):
        i_c = input[i, j]
        i_nbh = nbh(input, i, j)
        # cells adagent to the current one
        i_direct_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 0), (-1, 0), (0, 1), (0, -1)}}
        i_indirect_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 1), (-1, -1), (-1, 1), (1, -1)}}

        is_top_b, is_bottom_b = i == 0, i == input.shape[0] - 1
        is_left_b, is_right_b = j == 0, j == input.shape[1] - 1
        is_b = is_top_b or is_bottom_b or is_left_b or is_right_b

        if i_c > 0:
            output[i, j] = i_c

        for rule in rules:

            if i_c in rule['ignore_colors']:
                continue

            if rule['type'] == 'copy_color_by_direction':
                if rule['direction'] == 'bottom' or rule['direction'] == 'everywhere':
                    if not is_top_b and input[i - 1, j] in rule['copy_color'] and \
                            (i == 1 or input[i - 2, j] == rule['look_back_color']):
                        output[i, j] = input[i - 1, j]
                        break

                if rule['direction'] == 'top' or rule['direction'] == 'everywhere':
                    if not is_bottom_b and input[i + 1, j] in rule['copy_color'] and \
                            (i == input.shape[0] - 2 or input[i + 2, j] == rule['look_back_color']):
                        output[i, j] = input[i + 1, j]
                        break

                if rule['direction'] == 'right' or rule['direction'] == 'everywhere':
                    if not is_left_b and input[i, j - 1] in rule['copy_color'] and \
                            (j == 1 or input[i, j - 2] == rule['look_back_color']):
                        output[i, j] = input[i, j - 1]
                        break

                if rule['direction'] == 'left' or rule['direction'] == 'everywhere':
                    if not is_right_b and input[i, j + 1] in rule['copy_color'] and \
                            (j == input.shape[1] - 2 or input[i, j + 2] == rule['look_back_color']):
                        output[i, j] = input[i, j + 1]
                        break
            elif rule['type'] == 'corner_check':
                color_nbh = rule['nbh_check_colors']
                sum_nbh = 3
                out_nbh = rule['nbh_check_out']

                i_uplecorner_nbh = {k: v for k, v in i_nbh.items() if k in {(-1, -1), (-1, 0), (0, -1)}}
                i_upricorner_nbh = {k: v for k, v in i_nbh.items() if k in {(-1, 1), (-1, 0), (0, 1)}}
                i_dolecorner_nbh = {k: v for k, v in i_nbh.items() if k in {(1, -1), (1, 0), (0, -1)}}
                i_doricorner_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 1), (1, 0), (0, 1)}}
                if sum(1 for v in i_nbh.values() if v in color_nbh) < 3:
                    continue
                did_something = False
                for corner_idx in [i_uplecorner_nbh, i_upricorner_nbh, i_dolecorner_nbh, i_doricorner_nbh]:
                    for color in color_nbh:
                        if sum(1 for v in corner_idx.values() if v == color) == sum_nbh:
                            output[i, j] = out_nbh
                            did_something = True
                            break
                    if did_something:
                        break
                if did_something:
                    break


            elif rule['type'] == 'nbh_check':
                color_nbh = rule['nbh_check_colors']
                sum_nbh = rule['nbh_check_sum']
                out_nbh = rule['nbh_check_out']

                proper_nbhs = i_nbh.values()

                if sum(1 for v in proper_nbhs if v in color_nbh) > sum_nbh:
                    output[i, j] = out_nbh
                    break

            elif rule['type'] == 'direct_check':
                color_nbh = rule['nbh_check_colors']
                sum_nbh = rule['nbh_check_sum']
                out_nbh = rule['nbh_check_out']

                proper_nbhs = i_direct_nbh.values()

                if sum(1 for v in proper_nbhs if v in color_nbh) > sum_nbh:
                    output[i, j] = out_nbh
                    break

            elif rule['type'] == 'indirect_check':
                color_nbh = rule['nbh_check_colors']
                sum_nbh = rule['nbh_check_sum']
                out_nbh = rule['nbh_check_out']

                proper_nbhs = i_indirect_nbh.values()

                if sum(1 for v in proper_nbhs if v in color_nbh) > sum_nbh:
                    output[i, j] = out_nbh
                    break


            elif rule['type'] == 'color_distribution':
                directions = ['top', 'bottom', 'left', 'right', 'top_left', 'bottom_left', 'top_right', 'bottom_right']
                not_border_conditions = \
                    [
                        not is_top_b,
                        not is_bottom_b,
                        not is_left_b,
                        not is_right_b,
                        not is_top_b and not is_left_b,
                        not is_bottom_b and not is_left_b,
                        not is_top_b and not is_right_b,
                        not is_bottom_b and not is_right_b
                    ]
                index_from = \
                    [
                        (i - 1, j),
                        (i + 1, j),
                        (i, j - 1),
                        (i, j + 1),
                        (i - 1, j - 1),
                        (i + 1, j - 1),
                        (i - 1, j + 1),
                        (i + 1, j + 1)
                    ]

                did_something = False
                for i_dir, direction in enumerate(directions):
                    if rule['direction'] == direction:
                        if not_border_conditions[i_dir]:
                            if (rule['check_in_empty'] == 1 and input[index_from[i_dir]] > 0) or \
                                    (rule['check_in_empty'] == 0 and input[index_from[i_dir]] == rule['color_in']):
                                output[i, j] = rule['color_out']
                                did_something = True
                                break
                if did_something:
                    break

    return output, hidden_o

def get_connectivity_info(color: np.array, ignore_black = False, von_neumann_only = False, edge_for_difcolors = False):

    # UnionFind structure allows us to detect all connected areas in a linear time.
    class UnionFind:
        def __init__(self) -> None:
            self.area = np.ones(color.size)
            self.parent = np.arange(color.size)
        def find(self, x: int) -> int:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, u: int, v: int) -> None:
            root_u, root_v = self.find(u), self.find(v)
            if root_u != root_v:
                area_u, area_v = self.area[root_u], self.area[root_v]
                if area_u < area_v:
                    root_u, root_v = root_v, root_u
                self.parent[root_v] = root_u
                self.area[root_u] = area_u + area_v

    union_find = UnionFind()
    neighbours = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    if not von_neumann_only:
        neighbours.extend([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    nrows, ncols = color.shape
    for i in range(nrows):
        for j in range(ncols):
            for s, t in neighbours:
                u, v = i + s, j + t
                if u >= 0 and u < nrows and v >= 0 and v < ncols and \
                        (color[u, v] == color[i, j] or (edge_for_difcolors and (color[u, v]>0) == (color[i, j]>0))):
                    union_find.union(u * ncols + v, i * ncols + j)
    # for every cell: write down the area of its corresponding area
    communities = defaultdict(list)
    for i, j in product(range(nrows), range(ncols)):
        if not ignore_black or color[i, j] > 0:
            communities[union_find.find(i * ncols + j)].append((i, j))
    # the result is always sorted for consistency
    communities = sorted(communities.values(), key = lambda area: (len(area), area))
    return communities


def get_graph_communities(im, ignore_black=False):
    G = nx.Graph()
    I, J = im.shape
    for i in range(I):
        for j in range(J):
            if ignore_black and im[i, j] == 0:
                continue

            G.add_node((i, j))

            edges = []

            if j >= 1:
                if im[i, j] == im[i, j - 1]:
                    edges.append(((i, j), (i, j - 1)))

            if j < J - 1:
                if im[i, j] == im[i, j + 1]:
                    edges.append(((i, j), (i, j + 1)))

            if i >= 1:
                if im[i, j] == im[i - 1, j]:
                    edges.append(((i, j), (i - 1, j)))

                if j >= 1:
                    if im[i, j] == im[i - 1, j - 1]:
                        edges.append(((i, j), (i - 1, j - 1)))

                if j < J - 1:
                    if im[i, j] == im[i - 1, j + 1]:
                        edges.append(((i, j), (i - 1, j + 1)))

            if i < I - 1:
                if im[i, j] == im[i + 1, j]:
                    edges.append(((i, j), (i + 1, j)))

                if j >= 1:
                    if im[i, j] == im[i + 1, j - 1]:
                        edges.append(((i, j), (i + 1, j - 1)))
                if j < J - 1:
                    if im[i, j] == im[i + 1, j + 1]:
                        edges.append(((i, j), (i + 1, j + 1)))

            G.add_edges_from(edges)

    communities = list(nx.community.k_clique_communities(G, 2))

    communities = [list(com) for com in communities]
    for i in range(I):
        for j in range(J):
            i_nbh = nbh(im, i, j)
            if sum(1 for v in i_nbh.values() if v == im[i, j]) == 0:
                communities.append([(i, j)])

    return communities


def apply_rule(input, hidden_i, rule):
    output = np.zeros_like(input, dtype=int)
    #     print(type(input))
    #     print(input.shape)
    hidden = np.zeros_like(input)
    output[:, :] = input[:, :]
    if rule['type'] == 'macro_multiply_k':
        output = np.tile(output, rule['k'])
    elif rule['type'] == 'flip':
        if rule['how'] == 'ver':
            output = output[::-1, :]
        elif rule['how'] == 'hor':
            output = output[:, ::-1]

    elif rule['type'] == 'reduce':
        skip_row = np.zeros(input.shape[0])

        for i in range(1, input.shape[0]):
            skip_row[i] = (input[i] == input[i-1]).all() or (input[i] == rule['skip_color']).all()

        if (input[0] == rule['skip_color']).all():
            skip_row[0] = 1

        if np.sum(skip_row==0)>0:
            output = input[skip_row == 0]

        skip_column = np.zeros(input.shape[1])

        for i in range(1, input.shape[1]):
            skip_column[i] = (input[:, i] == input[:, i-1]).all() or (input[:, i] == rule['skip_color']).all()

        if (input[:, 0] == rule['skip_color']).all():
            skip_column[0] = 1

        if np.sum(skip_column==0)>0:
            output = output[:, skip_column == 0]


    elif rule['type'] == 'rotate':
        output = np.rot90(output, rule['rotations_count'])

    elif rule['type'] == 'micro_multiply_by':
        if rule['how_many'] == 'size':
            k = output.shape[0]
        else:
            k = rule['how_many']
        output = np.repeat(output, k, axis=0)
        output = np.repeat(output, k, axis=1)

    elif rule['type'] == 'macro_multiply_by':
        if rule['how_many'] == 'both':
            k = (2, 2)
        elif rule['how_many'] == 'hor':
            k = (1, 2)
        elif rule['how_many'] == 'ver':
            k = (2, 1)
        output = np.tile(output, k)
        if input.shape[0] == input.shape[1]:
            for i in range(k[0]):
                for j in range(k[1]):
                    sub = output[i * input.shape[0]: (i + 1) * input.shape[0],
                          j * input.shape[1]: (j + 1) * input.shape[1]]
                    sub_rotated = np.rot90(sub, rule['rotates'][i * 2 + j])
                    output[i * input.shape[0]: (i + 1) * input.shape[0],
                    j * input.shape[1]: (j + 1) * input.shape[1]] = sub_rotated
        for i in range(k[0]):
            for j in range(k[1]):
                sub = output[i * input.shape[0]: (i + 1) * input.shape[0], j * input.shape[1]: (j + 1) * input.shape[1]]
                if 'ver' in rule['flips'][i * 2 + j]:
                    sub = sub[::-1, :]
                if 'hor' in rule['flips'][i * 2 + j]:
                    sub = sub[:, ::-1]
                output[i * input.shape[0]: (i + 1) * input.shape[0], j * input.shape[1]: (j + 1) * input.shape[1]] = sub

    elif rule['type'] == 'distribute_from_border':
        hidden = np.zeros_like(input)
        for i in range(1, input.shape[0] - 1):
            if output[i, 0] in rule['colors']:
                if not output[i, input.shape[1] - 1] in rule['colors'] or output[i, input.shape[1] - 1] == output[i, 0]:
                    output[i] = output[i, 0]

        for j in range(1, input.shape[1] - 1):
            if output[0, j] in rule['colors']:
                if not output[input.shape[0] - 1, j] in rule['colors'] or output[input.shape[0] - 1, j] == output[0, j]:
                    output[:, j] = output[0, j]

    elif rule['type'] == 'color_for_inners':
        hidden = np.zeros_like(input)
        changed = 1
        while changed == 1:
            changed = 0
            for i, j in product(range(input.shape[0]), range(input.shape[1])):
                i_c = input[i, j]

                if i_c > 0 or hidden[i, j] == 1:
                    continue

                if i == 0 or i == input.shape[0] - 1 or j == 0 or j == input.shape[1] - 1:
                    hidden[i, j] = 1
                    changed = 1
                    continue

                i_nbh = nbh(hidden, i, j)
                # cells adagent to the current one
                i_direct_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 0), (-1, 0), (0, 1), (0, -1)}}

                if sum(1 for v in i_direct_nbh.values() if v == 1) > 0:
                    hidden[i, j] = 1
                    changed = 1
        output[((hidden == 0).astype(np.int) * (input == 0).astype(np.int)) == 1] = rule['color_out']
        hidden = np.copy(hidden)

    elif rule['type'] == 'draw_lines':
        hidden = np.zeros_like(input)
        if rule['direction'] == 'everywhere':
            directions = ['top', 'bottom', 'left', 'right', 'top_left', 'bottom_left', 'top_right', 'bottom_right']
        elif rule['direction'] == 'horizontal':
            directions = ['left', 'right']
        elif rule['direction'] == 'vertical':
            directions = ['top', 'bottom']
        elif rule['direction'] == 'horver':
            directions = ['top', 'bottom', 'left', 'right']
        elif rule['direction'] == 'diagonal':
            directions = ['top_left', 'bottom_left', 'top_right', 'bottom_right']
        else:
            directions = [rule['direction']]

        possible_directions = ['top', 'bottom', 'left', 'right',
                               'top_left', 'bottom_left', 'top_right', 'bottom_right']

        index_change = \
            [
                [-1, 0],
                [1, 0],
                (0, -1),
                (0, 1),
                (-1, -1),
                (+1, -1),
                (-1, +1),
                (+1, +1)
            ]
        for i_dir, direction in enumerate(possible_directions):
            if direction in directions:
                idx_ch = index_change[i_dir]
                for i in range(input.shape[0]):
                    for j in range(input.shape[1]):
                        if input[i, j] == rule['start_by_color']:
                            tmp_i = i + idx_ch[0]
                            tmp_j = j + idx_ch[1]
                            while 0 <= tmp_i < input.shape[0] and \
                                    0 <= tmp_j < input.shape[1] and \
                                    input[tmp_i, tmp_j] == rule['not_stop_by_color']:
                                output[tmp_i, tmp_j] = rule['with_color']
                                tmp_i += idx_ch[0]
                                tmp_j += idx_ch[1]

    elif rule['type'] == 'draw_line_to':
        hidden = np.zeros_like(input)

        index_change = \
            [
                [-1, 0],
                [1, 0],
                (0, -1),
                (0, 1),
            ]
        for i, j in product(range(input.shape[0]), range(input.shape[1])):
            if input[i, j] != rule['start_by_color']:
                continue

            number_0 = np.sum(output[:i] == rule['direction_color'])
            number_1 = np.sum(output[(i + 1):] == rule['direction_color'])
            number_2 = np.sum(output[:, :j] == rule['direction_color'])
            number_3 = np.sum(output[:, (j + 1):] == rule['direction_color'])
            i_dir = np.argmax([number_0, number_1, number_2, number_3])
            # print([number_0, number_1, number_2, number_3])
            # 1/0

            idx_ch = index_change[i_dir]
            tmp_i = i + idx_ch[0]
            tmp_j = j + idx_ch[1]
            while 0 <= tmp_i < input.shape[0] and \
                                    0 <= tmp_j < input.shape[1] and \
                    (input[tmp_i, tmp_j] in [rule['not_stop_by_color'], rule['not_stop_by_color_and_skip']]):

                skip_color = rule['not_stop_by_color_and_skip']
                if skip_color == 0 or input[tmp_i, tmp_j] != skip_color:
                    output[tmp_i, tmp_j] = rule['with_color']
                tmp_i += idx_ch[0]
                tmp_j += idx_ch[1]

    elif rule['type'] == 'distribute_colors':

        non_zero_rows = []
        non_zero_columns = []
        color_for_row = np.zeros(input.shape[0])
        color_for_column = np.zeros(input.shape[1])

        for i in range(input.shape[0]):
            row = input[i]
            colors, counts = np.unique(row, return_counts=True)
            good_colors = np.array([c in rule['colors'] for c in colors])
            if not good_colors.any():
                continue

            colors = colors[good_colors]
            counts = counts[good_colors]

            best_color = colors[np.argmax(counts)]
            color_for_row[i] = best_color
            non_zero_rows.append(i)

        for j in range(input.shape[1]):
            row = input[:, j]
            colors, counts = np.unique(row, return_counts=True)
            good_colors = np.array([c in rule['colors'] for c in colors])
            if not good_colors.any():
                continue

            colors = colors[good_colors]
            counts = counts[good_colors]

            best_color = colors[np.argmax(counts)]
            color_for_column[j] = best_color
            non_zero_columns.append(j)

        if rule['horizontally'] == 1:
            for i in non_zero_rows:
                output[i] = color_for_row[i]

        if rule['vertically'] == 1:
            for j in non_zero_columns:
                output[:, j] = color_for_column[j]

        for i in non_zero_rows:
            for j in non_zero_columns:
                if input[i, j] == 0:
                    output[i, j] = rule['intersect']
        hidden = np.copy(hidden_i)

    elif rule['type'] == 'unity':
        hidden = np.copy(hidden_i)
        if rule['mode'] == 'vertical':
            for j in range(input.shape[1]):
                last_color_now = np.zeros(10, dtype=np.int) - 1
                for i in range(input.shape[0]):
                    if not input[i, j] in rule['ignore_colors'] and last_color_now[input[i, j]] >= 0:
                        if rule['with_color'] == 0:
                            output[(last_color_now[input[i, j]] + 1):i, j] = input[i, j]
                        else:
                            output[(last_color_now[input[i, j]] + 1):i, j] = rule['with_color']
                        last_color_now[input[i, j]] = i
                    elif not input[i, j] in rule['ignore_colors']:
                        last_color_now[input[i, j]] = i


        elif rule['mode'] == 'horizontal':
            for i in range(input.shape[0]):
                last_color_now = np.zeros(10, dtype=np.int) - 1
                for j in range(input.shape[1]):
                    if not input[i, j] in rule['ignore_colors'] and last_color_now[input[i, j]] >= 0:
                        if rule['with_color'] == 0:
                            output[i, (last_color_now[input[i, j]] + 1):j] = input[i, j]
                        else:
                            output[i, (last_color_now[input[i, j]] + 1):j] = rule['with_color']
                        last_color_now[input[i, j]] = j
                    elif not input[i, j] in rule['ignore_colors']:
                        last_color_now[input[i, j]] = j

        elif rule['mode'] == 'horver':
            for j in range(input.shape[1]):
                last_color_now = np.zeros(10, dtype=np.int) - 1
                for i in range(input.shape[0]):
                    if not input[i, j] in rule['ignore_colors'] and last_color_now[input[i, j]] >= 0:
                        if rule['with_color'] == 0:
                            output[(last_color_now[input[i, j]] + 1):i, j] = input[i, j]
                        else:
                            output[(last_color_now[input[i, j]] + 1):i, j] = rule['with_color']
                        last_color_now[input[i, j]] = i
                    elif not input[i, j] in rule['ignore_colors']:
                        last_color_now[input[i, j]] = i

            for i in range(input.shape[0]):
                last_color_now = np.zeros(10, dtype=np.int) - 1
                for j in range(input.shape[1]):
                    if not input[i, j] in rule['ignore_colors'] and last_color_now[input[i, j]] >= 0:
                        if rule['with_color'] == 0:
                            output[i, (last_color_now[input[i, j]] + 1):j] = input[i, j]
                        else:
                            output[i, (last_color_now[input[i, j]] + 1):j] = rule['with_color']
                        last_color_now[input[i, j]] = j
                    elif not input[i, j] in rule['ignore_colors']:
                        last_color_now[input[i, j]] = j

        elif rule['mode'] == 'diagonal':
            for diag_id in range(-input.shape[0] - 1, input.shape[1] + 1):
                last_color_now_x = np.zeros(10, dtype=np.int) - 1
                last_color_now_y = np.zeros(10, dtype=np.int) - 1
                for i, j in zip(np.arange(input.shape[0]), diag_id + np.arange(input.shape[0])):
                    if 0 <= i < input.shape[0] and 0 <= j < input.shape[1]:
                        if not input[i, j] in rule['ignore_colors'] and last_color_now_x[input[i, j]] >= 0:

                            if rule['with_color'] == 0:
                                output[np.arange(last_color_now_x[input[i, j]] + 1, i), np.arange(
                                    last_color_now_y[input[i, j]] + 1, j)] = input[i, j]
                            else:
                                output[np.arange(last_color_now_x[input[i, j]] + 1, i), np.arange(
                                    last_color_now_y[input[i, j]] + 1, j)] = rule[
                                    'with_color']
                            last_color_now_x[input[i, j]] = i
                            last_color_now_y[input[i, j]] = j

                        elif not input[i, j] in rule['ignore_colors']:
                            last_color_now_x[input[i, j]] = i
                            last_color_now_y[input[i, j]] = j

            reflected_input = input[:, ::-1]
            output = output[:, ::-1]
            for diag_id in range(-reflected_input.shape[0] - 1, reflected_input.shape[1] + 1):
                last_color_now_x = np.zeros(10, dtype=np.int) - 1
                last_color_now_y = np.zeros(10, dtype=np.int) - 1
                for i, j in zip(np.arange(reflected_input.shape[0]), diag_id + np.arange(reflected_input.shape[0])):
                    if 0 <= i < reflected_input.shape[0] and 0 <= j < reflected_input.shape[1]:
                        if not reflected_input[i, j] in rule['ignore_colors'] and last_color_now_x[
                            reflected_input[i, j]] >= 0:

                            if rule['with_color'] == 0:
                                output[np.arange(last_color_now_x[reflected_input[i, j]] + 1, i), np.arange(
                                    last_color_now_y[reflected_input[i, j]] + 1, j)] = reflected_input[i, j]
                            else:
                                output[np.arange(last_color_now_x[reflected_input[i, j]] + 1, i), np.arange(
                                    last_color_now_y[reflected_input[i, j]] + 1, j)] = rule[
                                    'with_color']
                            last_color_now_x[reflected_input[i, j]] = i
                            last_color_now_y[reflected_input[i, j]] = j

                        elif not reflected_input[i, j] in rule['ignore_colors']:
                            last_color_now_x[reflected_input[i, j]] = i
                            last_color_now_y[reflected_input[i, j]] = j
            output = output[:, ::-1]
    elif rule['type'] == 'split_by_H':
        hidden = np.copy(hidden_i)

        if output.shape[0] >= 2:
            part1 = output[:int(np.floor(output.shape[0] / 2))]
            part2 = output[int(np.ceil(output.shape[0] / 2)):]

            output = np.zeros_like(part1)
            if rule['merge_rule'] == 'or':
                output[part1 > 0] = part1[part1 > 0]
                output[part2 > 0] = part2[part2 > 0]
            elif rule['merge_rule'] == 'equal':
                idx = np.logical_and(np.logical_and(part1 > 0, part2 > 0), part1 == part2)
                output[idx] = part1[idx]
            elif rule['merge_rule'] == 'and':
                idx = np.logical_and(part1 > 0, part2 > 0)
                output[idx] = part1[idx]
            elif rule['merge_rule'] == 'xor':
                idx = np.logical_xor(part1 > 0, part2 > 0)
                output[idx] = part1[idx]

    elif rule['type'] == 'split_by_W':
        hidden = np.copy(hidden_i)

        if output.shape[1] >= 2:
            part1 = output[:, :int(np.floor(output.shape[1] / 2))]
            part2 = output[:, int(np.ceil(output.shape[1] / 2)):]
            output = np.zeros_like(part1)
            if rule['merge_rule'] == 'or':
                output[part1 > 0] = part1[part1 > 0]
                output[part2 > 0] = part2[part2 > 0]
            elif rule['merge_rule'] == 'equal':
                idx = np.logical_and(np.logical_and(part1 > 0, part2 > 0), part1 == part2)
                output[idx] = part1[idx]
            elif rule['merge_rule'] == 'and':
                idx = np.logical_and(part1 > 0, part2 > 0)
                output[idx] = part1[idx]
            elif rule['merge_rule'] == 'xor':
                idx = np.logical_xor(part1 > 0, part2 > 0)
                output[idx] = part1[idx]


    elif rule['type'] == 'map_color':
        hidden = np.copy(hidden_i)
        output[output == rule['color_in']] = rule['color_out']
    elif rule['type'] == 'crop_empty':

        hidden = np.copy(hidden_i)

        nonzerosi = np.max((output != 0).astype(np.int), axis=1)
        nonzerosj = np.max((output != 0).astype(np.int), axis=0)
        #         print(nonzerosi)
        #         print(nonzerosj)
        if np.max(nonzerosi) == 0 or np.max(nonzerosj) == 0:
            output = output * 0
        else:
            mini = np.min(np.arange(output.shape[0])[nonzerosi == 1])
            maxi = np.max(np.arange(output.shape[0])[nonzerosi == 1])
            minj = np.min(np.arange(output.shape[1])[nonzerosj == 1])
            maxj = np.max(np.arange(output.shape[1])[nonzerosj == 1])

            output = output[mini:(maxi + 1), minj:(maxj + 1)]

    elif rule['type'] == 'crop_figure':
        hidden = np.copy(hidden_i)

        communities = get_connectivity_info(output, ignore_black=True, edge_for_difcolors=rule['dif_c_edge'])
        if len(communities) == 0:
            output = np.zeros_like(output)
        else:
            if rule['mode'] == 'biggest':
                biggest = list(communities[np.argmax([len(list(com)) for com in communities])])
            else:
                biggest = list(communities[np.argmin([len(list(com)) for com in communities])])
            biggest = np.array(biggest)
            min_bx = np.min(biggest[:, 0])
            min_by = np.min(biggest[:, 1])
            biggest[:, 0] -= min_bx
            biggest[:, 1] -= min_by
            output = np.zeros((np.max(biggest[:, 0]) + 1, np.max(biggest[:, 1]) + 1), dtype=np.int)
            for i in range(biggest.shape[0]):
                output[tuple(biggest[i])] = input[(min_bx + biggest[i][0], min_by + biggest[i][1])]

    elif rule['type'] == 'make_holes':
        hidden = np.copy(hidden_i)

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                i_nbh = nbh(output, i, j)
                proper_nbhs = i_nbh.values()
                for color in range(1, 10):
                    if sum(1 for v in proper_nbhs if v == color) == 8:
                        output[i, j] = 0
                        break

    elif rule['type'] == 'gravity':
        changed_smth = 1
        hidden = np.copy(hidden_i)
        im = output

        if rule['gravity_type'] == 'figures':
            communities = get_connectivity_info(im, ignore_black=True)

        else:
            communities = []
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    if output[i, j] > 0:
                        communities.append([[i, j]])

        directions = []

        for com in communities:
            community = list(com)
            color_fig = output[community[0][0], community[0][1]]
            if rule['look_at_what_to_move'] == 1 and color_fig != rule['color_what']:
                directions.append('None')
                continue

            xs = [p[0] for p in community]
            ys = [p[1] for p in community]

            if rule['direction_type'] == 'border':
                direction = rule['direction_border']
            elif rule['direction_type'] == 'color':
                color = rule['direction_color']
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                number_0 = np.sum(output[:xmin] == color)
                number_1 = np.sum(output[(xmax + 1):] == color)
                number_2 = np.sum(output[:, :ymin] == color)
                number_3 = np.sum(output[:, (ymax + 1):] == color)
                direction = ['top', 'bottom', 'left', 'right'][np.argmax([number_0, number_1, number_2, number_3])]

            directions.append(direction)

        already_moved = np.zeros(len(communities))
        while changed_smth > 0:

            changed_smth = 0

            for i, com in enumerate(communities):
                community = list(com)
                color_fig = output[community[0][0], community[0][1]]
                xs = [p[0] for p in community]
                ys = [p[1] for p in community]

                direction = directions[i]
                if direction == 'top':
                    toper = np.array([[p[0] - 1, p[1]] for p in community if (p[0] - 1, p[1]) not in community])
                    xs = np.array([p[0] for p in toper])
                    ys = np.array([p[1] for p in toper])
                    if np.min(xs) < 0:
                        continue

                    if (output[xs, ys] == 0).all() and (rule['steps_limit']==1 or already_moved[i]==0):
                        changed_smth = 1
                        already_moved[i]=1
                        com_xs = np.array([p[0] for p in community])
                        com_ys = np.array([p[1] for p in community])
                        output[com_xs, com_ys] = 0
                        output[com_xs - 1, com_ys] = color_fig
                        communities[i] = [(p[0] - 1, p[1]) for p in community]

                if direction == 'bottom':
                    toper = np.array([[p[0] + 1, p[1]] for p in community if (p[0] + 1, p[1]) not in community])
                    xs = np.array([p[0] for p in toper])
                    ys = np.array([p[1] for p in toper])

                    if np.max(xs) == input.shape[0]:
                        continue

                    if (output[xs, ys] == 0).all() and (rule['steps_limit']==1 or already_moved[i]==0):
                        changed_smth = 1
                        already_moved[i]=1
                        com_xs = np.array([p[0] for p in community])
                        com_ys = np.array([p[1] for p in community])

                        output[com_xs, com_ys] = 0
                        output[com_xs + 1, com_ys] = color_fig
                        communities[i] = [(p[0] + 1, p[1]) for p in community]

                if direction == 'left':
                    toper = np.array([[p[0], p[1] - 1] for p in community if (p[0], p[1] - 1) not in community])
                    xs = np.array([p[0] for p in toper])
                    ys = np.array([p[1] for p in toper])

                    if np.min(ys) < 0:
                        continue

                    if (output[xs, ys] == 0).all() and (rule['steps_limit']==1 or already_moved[i]==0):
                        changed_smth = 1
                        already_moved[i]=1
                        com_xs = np.array([p[0] for p in community])
                        com_ys = np.array([p[1] for p in community])
                        output[com_xs, com_ys] = 0
                        output[com_xs, com_ys - 1] = color_fig
                        communities[i] = [(p[0], p[1] - 1) for p in community]

                if direction == 'right':
                    toper = np.array([[p[0], p[1] + 1] for p in community if (p[0], p[1] + 1) not in community])
                    xs = np.array([p[0] for p in toper])
                    ys = np.array([p[1] for p in toper])

                    if np.max(ys) == input.shape[1]:
                        continue

                    if (output[xs, ys] == 0).all() and (rule['steps_limit']==1 or already_moved[i]==0):
                        changed_smth = 1
                        already_moved[i]=1
                        com_xs = np.array([p[0] for p in community])
                        com_ys = np.array([p[1] for p in community])
                        output[com_xs, com_ys] = 0
                        output[com_xs, com_ys + 1] = color_fig
                        communities[i] = [(p[0], p[1] + 1) for p in community]


    return output, hidden


def compute_metrics(prediction_grid, answer_grid):
    n_metrics = 11

    def get_metrics(prediction, answer):
        prediction_empty = (prediction == 0).astype(np.int)
        answer_empty = (answer == 0).astype(np.int)

        right = (prediction == answer).astype(np.int)
        # empty_right = (prediction_empty == answer_empty).astype(np.int)
        #
        accuracy = np.mean(right)
        # accuracy_empty = np.mean(empty_right)
        # precision = 1 - np.mean((1 - prediction_empty) * (1 - right))
        # recall = 1 - np.mean((1 - answer_empty) * (1 - right))
        # precision_empty = 1 - np.mean((1 - prediction_empty) * (1 - empty_right))
        # recall_empty = 1 - np.mean((1 - answer_empty) * (1 - empty_right))
        # return [accuracy,
        #         accuracy_empty,
        #         precision, recall,
        #         precision_empty, recall_empty
        #         ][:n_metrics]
        color_rights = []
        for color in range(10):
            idx = answer != color
            # print(idx.astype(np.int))
            color_right = float((np.logical_or(idx, right).all() and not (prediction[idx]==color).any()))
            color_rights.append(color_right)
        #print(color_rights)
        #print(color_rights)
        #1/0
        # right = (prediction == answer).astype(np.int)
        # empty_right = (prediction_empty == answer_empty).astype(np.int)
        #
        # accuracy = np.mean(right)
        # accuracy_empty = np.mean(empty_right)
        # precision = 1 - np.mean((1 - prediction_empty) * (1 - right))
        # recall = 1 - np.mean((1 - answer_empty) * (1 - right))
        # precision_empty = 1 - np.mean((1 - prediction_empty) * (1 - empty_right))
        # recall_empty = 1 - np.mean((1 - answer_empty) * (1 - empty_right))
        return [accuracy] + color_rights

    #print(prediction_grid.shape, answer_grid.shape)
    if prediction_grid.shape == answer_grid.shape:
        # print(prediction_grid)
        # print(answer_grid)
        mets = get_metrics(prediction_grid, answer_grid) + [1]
        #print(mets)
        return mets

    # elif prediction_grid.shape[0] >= answer_grid.shape[0] and prediction_grid.shape[1] >= answer_grid.shape[1]:
    #     metrics = np.zeros((prediction_grid.shape[0] - answer_grid.shape[0] + 1,
    #                         prediction_grid.shape[1] - answer_grid.shape[1] + 1, n_metrics))
    #     for i in range(prediction_grid.shape[0] - answer_grid.shape[0] + 1):
    #         for j in range(prediction_grid.shape[1] - answer_grid.shape[1] + 1):
    #             prediction = prediction_grid[i:(i + answer_grid.shape[0]), j:(j + answer_grid.shape[1])]
    #             metrics[i, j] = get_metrics(prediction, answer_grid)
    #
    #     maxi, maxj = np.unravel_index(metrics[:, :, 0].argmax(), metrics[:, :, 0].shape)
    #     # mean_metrics = list(np.mean(np.mean(metrics, axis=0), axis=0)/2 + np.array(metrics[maxi, maxj])/2)
    #     size_proportion = answer_grid.shape[0] * answer_grid.shape[1] / prediction_grid.shape[0] / \
    #                       prediction_grid.shape[1]
    #     metrics = metrics[maxi, maxj]
    #     return list(metrics) + [size_proportion]
    #
    # elif prediction_grid.shape[0] <= answer_grid.shape[0] and prediction_grid.shape[1] <= answer_grid.shape[1]:
    #     metrics = np.zeros((answer_grid.shape[0] - prediction_grid.shape[0] + 1,
    #                         answer_grid.shape[1] - prediction_grid.shape[1] + 1, n_metrics))
    #     for i in range(answer_grid.shape[0] - prediction_grid.shape[0] + 1):
    #         for j in range(answer_grid.shape[1] - prediction_grid.shape[1] + 1):
    #             answer = answer_grid[i:(i + prediction_grid.shape[0]), j:(j + prediction_grid.shape[1])]
    #             metrics[i, j] = get_metrics(prediction_grid, answer)
    #
    #     maxi, maxj = np.unravel_index(metrics[:, :, 0].argmax(), metrics[:, :, 0].shape)
    #    # mean_metrics = list(np.mean(np.mean(metrics, axis=0), axis=0)/2 + np.array(metrics[maxi, maxj])/2)
    #     size_proportion = answer_grid.shape[0] * answer_grid.shape[1] / prediction_grid.shape[0] / \
    #                       prediction_grid.shape[1]
    #     metrics = metrics[maxi, maxj]
    #     return list(metrics) + [1/size_proportion]

    # elif prediction_grid.shape[0] >= answer_grid.shape[0] and prediction_grid.shape[1] >= answer_grid.shape[1]:
    #     maxi, maxj = 0, 0
    #     maxcommon = 0
    #
    #     for i in range(prediction_grid.shape[0] - answer_grid.shape[0] + 1):
    #         for j in range(prediction_grid.shape[1] - answer_grid.shape[1] + 1):
    #             for i_check, j_check in product(range(answer_grid.shape[0]), range(answer_grid.shape[1])):
    #                 if prediction_grid[i + i_check, j + j_check] != answer_grid[i_check, j_check]:
    #                     common = i_check * j_check
    #                     break
    #                 if i_check == answer_grid.shape[0] - 1 and j_check == answer_grid.shape[1] - 1:
    #                     common = i_check * j_check
    #
    #             if common > maxcommon:
    #                 maxi = i
    #                 maxj = j
    #                 maxcommon = common
    #                 if common == answer_grid.shape[0] * answer_grid.shape[1]:
    #                     break
    #
    #     metrics = get_metrics(prediction_grid[maxi:(maxi + answer_grid.shape[0]),
    #                           maxj:(maxj + answer_grid.shape[1])], answer_grid)
    #
    #     modified_pred = np.zeros_like(prediction_grid)
    #     modified_pred[:] = prediction_grid[:]
    #     modified_pred[maxi:(maxi + answer_grid.shape[0]), maxj:(maxj + answer_grid.shape[1])] = 0
    #     size_proportion =  answer_grid.shape[0] * answer_grid.shape[1] / prediction_grid.shape[0] / prediction_grid.shape[1]
    #     #print(np.mean(modified_pred==0))
    #     return list(size_proportion*np.array(metrics)) + [1.0]
    #
    # elif prediction_grid.shape[0] <= answer_grid.shape[0] and prediction_grid.shape[1] <= answer_grid.shape[1]:
    #     maxi, maxj = 0, 0
    #     maxcommon = 0
    #
    #     for i in range(answer_grid.shape[0] - prediction_grid.shape[0] + 1):
    #         for j in range(answer_grid.shape[1] - prediction_grid.shape[1] + 1):
    #             for i_check, j_check in product(range(prediction_grid.shape[0]), range(prediction_grid.shape[1])):
    #                 #print(i_check, j_check)
    #                 if answer_grid[i + i_check, j + j_check] != prediction_grid[i_check, j_check]:
    #                     common = i_check * j_check
    #                     break
    #                 if i_check == prediction_grid.shape[0] - 1 and j_check == prediction_grid.shape[1] - 1:
    #                     common = i_check * j_check
    #
    #             if common > maxcommon:
    #                 maxi = i
    #                 maxj = j
    #                 maxcommon = common
    #                 if common == prediction_grid.shape[0] * prediction_grid.shape[1]:
    #                     break
    #
    #     metrics = get_metrics(answer_grid[maxi:(maxi + prediction_grid.shape[0]),
    #                           maxj:(maxj + prediction_grid.shape[1])], prediction_grid)
    #
    #     modified_pred = np.zeros_like(answer_grid)
    #     modified_pred[:] = answer_grid[:]
    #     modified_pred[maxi:(maxi + prediction_grid.shape[0]), maxj:(maxj + prediction_grid.shape[1])] = 0
    #     size_proportion = prediction_grid.shape[0] * prediction_grid.shape[1] / answer_grid.shape[0] / answer_grid.shape[1]
    #     return list(size_proportion*np.array(metrics)) + [1.0]


    return list(np.array(get_metrics(answer_grid, answer_grid)) * 0) + [0]


def validate_automata(task_global, params, n_iter_max, n_hidden):
    def validate(task):
        inp = task['input']

        out = trace_param_automata(inp, params, n_iter_max, n_hidden)[-1][0]

        metrics = compute_metrics(out, task['output'])

        return metrics

    metrics = []
    for task in task_global['train']:
        metrics.append(validate(task))

    mean_metrics = list(np.round(np.mean(metrics, axis=0), 3))
    min_metrics = list(np.round(np.min(metrics, axis=0), 3))

    return tuple(mean_metrics + list(np.array(metrics)[:, 0].reshape(-1)))#tuple(mean_metrics + min_metrics)


def product_better(a, b):
    """ Return True iff the two tuples a and b respect a<b for the partial order. """
    a = np.array(a)
    b = np.array(b)
    return (np.array(a) >= np.array(b)).all() and (np.array(a) > np.array(b)).any()



def generate_random_ca(all_colors, best_candidates, temp, config, length=1):
    rules = []
    for _ in range(length):
        rules.append(get_random_ca_rule(all_colors, best_candidates, temp, config))
    return rules

def generate_random_global(all_colors, best_candidates, temp, config, length=1):
    rules = []
    for _ in range(length):
        rules.append(get_random_global_rule(all_colors, best_candidates, temp, config))
    return rules

def generate_population(all_colors, config, size=64, length=1):
    population = []
    for i in range(size):
        split_rule = get_random_split_rule(all_colors, {}, 0, config)
        merge_rule = get_random_merge_rule(all_colors, {}, 0, config)
        global_rules = generate_random_global(all_colors, {}, 0, config, np.random.choice(2, p=[0.2, 0.8]))
        ca_rules = generate_random_ca(all_colors, {}, 0, config, np.random.choice(2, p=[0.2, 0.8]))
        population.append([global_rules, ca_rules, split_rule, merge_rule])

    return population

from pathlib import Path
import json

train_path = data_path / 'training'
valid_path = data_path / 'evaluation'
test_path = data_path / 'test'
submission_path = data_path / 'public_submission.csv'


train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() }
valid_tasks = { task.stem: json.load(task.open()) for task in valid_path.iterdir() }
test_path = { task.stem: json.load(task.open()) for task in test_path.iterdir() }
train_task_ids = np.sort(list(train_tasks.keys()))
valid_task_ids = np.sort(list(valid_tasks.keys()))
test_task_ids = np.sort(list(test_path.keys()))

from functools import partial
from itertools import product
from sklearn.preprocessing import MinMaxScaler

def change_color(colors_in, colors_out, grid):
    out_grid = np.zeros_like(grid)
    out_grid[:] = grid[:]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for color_in, color_out in zip(colors_in, colors_out):
                if grid[i, j] == color_in:
                    out_grid[i, j] = color_out
                    break
    return out_grid


def reduce_grid(grid_rows, grid_columns, color, grid):
    out_grid = np.zeros((len(grid_rows), len(grid_columns)), dtype=np.int)
    for i, j in product(range(len(grid_rows)), range(len(grid_columns))):
        out_grid[i, j] = grid[grid_rows[i][0], grid_columns[j][0]]
    return out_grid


def unreduce_grid(line_rows, line_columns, n, m, grid_rows, grid_columns, color, grid):
    out_grid = np.zeros((n, m), dtype=np.int)
    for i in range(len(line_rows)):
        out_grid[line_rows[i]] = color

    for j in range(len(line_columns)):
        out_grid[:, line_columns[j]] = color

    for i, j in product(range(len(grid_rows)), range(len(grid_columns))):
        if grid[i, j] != 0:
            for i_gr_row in list(grid_rows[i]):
                for j_gr_col in list(grid_columns[j]):
                    out_grid[i_gr_row, j_gr_col] = grid[i, j]

    return out_grid


def get_color_features(input_grid):
    colors = np.unique(input_grid)
    colors_numbers = np.array([np.mean(input_grid == color) for color in colors]).reshape((-1, 1))

    # communities_1 = get_graph_communities(input_grid)
    #
    # communities_2 = get_connectivity_info(input_grid)
    #
    # communities_1 = sorted([sorted(com) for com in communities_1])
    # communities_2 = sorted([sorted(com) for com in communities_2])
    #
    # assert all((a == b) for a, b in zip(communities_1, communities_2))

    # colors_communities = [np.sum([input_grid[list(com)[0]] == color for com in communities]) / len(communities) for
    #                       color in colors]
    #colors_communities = np.array(colors_communities).reshape((-1, 1))

    colors_borders = np.array([np.mean(input_grid[0] == color) for color in colors]).reshape((-1, 1))
    colors_borders += np.array([np.mean(input_grid[-1] == color) for color in colors]).reshape((-1, 1))
    colors_borders += np.array([np.mean(input_grid[:, 0] == color) for color in colors]).reshape((-1, 1))
    colors_borders += np.array([np.mean(input_grid[:, -1] == color) for color in colors]).reshape((-1, 1))

    colors_borders /= np.sum(colors_borders)

    colors_features = np.concatenate([colors_numbers, colors_borders], axis=1)

    return colors_features, colors


def get_train_color_features(task):
    colors_in_train = []
    colors_in_each_train = []
    for uni_task in task['train']:
        inp = uni_task['input']
        colors_unique, color_numbers = np.unique(inp, return_counts=True)

        colors_in_train += list(colors_unique)
        colors_in_each_train.append(colors_unique)

    max_color_task = np.argmax([clrs.shape[0] for clrs in colors_in_each_train])
    colors = colors_in_each_train[max_color_task]
    input_grid = task['train'][max_color_task]['input']

    train_colors_features, _ = get_color_features(input_grid)

    scaler = MinMaxScaler()
    train_colors_features = scaler.fit_transform(train_colors_features)

    sums = np.sum(train_colors_features, axis=1)

    train_colors_features = train_colors_features[np.argsort(sums)[::-1]]

    return train_colors_features, scaler, np.unique(colors_in_train)


def build_mapping(task, config):
    reverse_functions = []
    for part in ['train', 'test']:
        for uni_task in task[part]:
            if part == 'test':
                reverse_functions.append({})

    if config['reduce_grid']:
        can_reduce_grid = True
        for uni_task in task['train']:
            if uni_task['input'].shape != uni_task['output'].shape:
                can_reduce_grid = False
                break

            inp = uni_task['input']

            colors_rows = []
            line_rows = []

            for i in range(inp.shape[0]):
                if (inp[i] == inp[i][0]).all():
                    colors_rows.append(inp[i][0])
                    line_rows.append(i)

            row_colors, row_counts = np.unique(colors_rows, return_counts=True)

            colors_columns = []
            line_columns = []

            for i in range(inp.shape[1]):
                if (inp[:, i] == inp[0, i]).all():
                    colors_columns.append(inp[0, i])
                    line_columns.append(i)

            column_colors, column_counts = np.unique(colors_columns, return_counts=True)

            if row_colors.shape[0] != 1 or column_colors.shape[0] != 1 or \
                    row_counts[0] < 2 or column_counts[0] < 2:

                can_reduce_grid = False
                break

            line_rows.append(inp.shape[0])
            line_rows = [-1] + line_rows

            line_columns.append(inp.shape[1])
            line_columns = [-1] + line_columns

            for i in range(len(line_rows) - 1):
                if (line_rows[i] + 1) < line_rows[i + 1]:
                    for j in range(len(line_columns) - 1):
                        if (line_columns[j] + 1) < line_columns[j + 1]:
                            color = inp[line_rows[i] + 1][line_columns[j] + 1]
                            if not (inp[(line_rows[i] + 1):(line_rows[i + 1]),
                                    (line_columns[j] + 1):(line_columns[j + 1])] == color).all():
                                can_reduce_grid = False
                                break

            for i in range(1, len(line_rows) - 1):
                if not (uni_task['input'][line_rows[i]] == uni_task['output'][line_rows[i]]).all():
                    can_reduce_grid = False
                    break

            for j in range(1, len(line_columns) - 1):
                if not (uni_task['input'][:, line_columns[j]] == uni_task['output'][:, line_columns[j]]).all():
                    can_reduce_grid = False
                    break

            if not can_reduce_grid:
                break

        if can_reduce_grid:
            for part in ['train', 'test']:
                for i_task, uni_task in enumerate(task[part]):
                    inp = uni_task['input']

                    colors_rows = []
                    line_rows = []

                    for i in range(inp.shape[0]):
                        if (inp[i] == inp[i][0]).all():
                            colors_rows.append(inp[i][0])
                            line_rows.append(i)

                    row_colors, row_counts = np.unique(colors_rows, return_counts=True)

                    colors_columns = []
                    line_columns = []

                    for i in range(inp.shape[1]):
                        if (inp[:, i] == inp[0, i]).all():
                            colors_columns.append(inp[0, i])
                            line_columns.append(i)

                    column_colors, column_counts = np.unique(colors_columns, return_counts=True)

                    line_rows.append(inp.shape[0])
                    line_rows = [-1] + line_rows

                    line_columns.append(inp.shape[1])
                    line_columns = [-1] + line_columns

                    grid_rows = []
                    grid_columns = []

                    for i in range(len(line_rows) - 1):
                        if (line_rows[i] + 1) < line_rows[i + 1]:
                            grid_rows.append(np.arange(line_rows[i] + 1, line_rows[i + 1]))

                    for j in range(len(line_columns) - 1):
                        if (line_columns[j] + 1) < line_columns[j + 1]:
                            grid_columns.append(np.arange(line_columns[j] + 1, line_columns[j + 1]))

                    uni_task['input'] = reduce_grid(grid_rows, grid_columns, row_colors[0], inp)
                    if part == 'train':
                        uni_task['output'] = reduce_grid(grid_rows, grid_columns, row_colors[0], uni_task['output'])
                    if part == 'test':
                        reverse_functions[i_task]['unreduce_grid'] = partial(unreduce_grid, line_rows[1:-1],
                                                                             line_columns[1:-1], inp.shape[0],
                                                                             inp.shape[1],
                                                                             grid_rows, grid_columns, row_colors[0])

    if config['map_color']:
        go_map_color = True
        train_colors_features, scaler, unique_train_colors = get_train_color_features(task)

        for uni_task in task['test']:
            inp = uni_task['input']
            colors_test = list(np.unique(inp))
            for color in colors_test:
                if not color in unique_train_colors:
                    go_map_color = True

        if go_map_color:
            colors_in_all = [[], []]
            colors_out_all = [[], []]

            for i_part, part in enumerate(['train', 'test']):
                for i_task, uni_task in enumerate(task[part]):

                    input_grid = uni_task['input']

                    colors_features, colors = get_color_features(input_grid)
                    proper_colors = list(np.arange(train_colors_features.shape[0]))
                    colors_features = scaler.transform(colors_features)
                    colors_in = []
                    colors_out = []
                    for i, color in enumerate(colors):
                        color_features = colors_features[i].reshape((1, -1))
                        distances = np.sum(np.power(train_colors_features - color_features, 2), axis=1)
                        closests = list(np.argsort(distances))
                        for closest in closests:
                            if closest in proper_colors:
                                proper_colors.remove(closest)
                                colors_in.append(color)
                                colors_out.append(closest)
                                break

                    if part == 'train':
                        colors_in_all[i_part].append(colors_in)
                        colors_out_all[i_part].append(colors_out)
                    if part == 'test':
                        colors_in_all[i_part].append(colors_out)
                        colors_out_all[i_part].append(colors_in)
                        reverse_functions[i_task]['train_colors_in'] = colors_out
                        reverse_functions[i_task]['train_colors_out'] = colors_in

            unique_test_colors = []

            for i_task, uni_task in enumerate(task['train']):

                output_grid = uni_task['output']
                colors = np.unique(output_grid)

                for color in colors:
                    if not color in unique_train_colors:
                        unique_test_colors.append(color)

            unique_test_colors = np.unique(unique_test_colors)
            colors_out = 9 - np.arange(unique_test_colors.shape[0])
            for part in ['train', 'test']:
                for i_task, uni_task in enumerate(task[part]):
                    if part == 'train':
                        uni_task['input'] = change_color(colors_in_all[0][i_task], colors_out_all[0][i_task],
                                                         uni_task['input'])
                        colors_in_all[0][i_task] += list(unique_test_colors)
                        colors_out_all[0][i_task] += list(colors_out)
                        uni_task['output'] = change_color(colors_in_all[0][i_task], colors_out_all[0][i_task],
                                                          uni_task['output'])
                    if part == 'test':
                        reverse_functions[i_task]['test_colors_in'] = list(colors_out)
                        reverse_functions[i_task]['test_colors_out'] = list(unique_test_colors)

    if config['find_wall']:
        for i_part, part in enumerate(['train', 'test']):
            for i_task, uni_task in enumerate(task[part]):

                input_grid = uni_task['input']

                colors_features, colors = get_color_features(input_grid)

                sums = np.sum(colors_features, axis=1)

                color_wall = colors[np.argsort(sums)[::-1][0]]
                #print(color_wall)
                if color_wall == 0:
                    continue

                colors_in = [0, color_wall]
                colors_out = [color_wall, 0]

                uni_task['input'] = change_color(colors_in, colors_out, input_grid)
                if part == 'train':
                    uni_task['output'] = change_color(colors_in, colors_out, uni_task['output'])
                if part == 'test':
                    reverse_functions[i_task]['return_wall'] = partial(change_color, colors_out,
                                                                       colors_in)

    return task, reverse_functions


def update_pool(task, best_candidates, candidate, num_params):
    start = time.time()
    score = validate_automata(task, candidate, 25, 1)
    is_uncomp = True
    updated_keys = False
    best_candidates_items = list(best_candidates.items())
    for best_score, best_candidates_score in best_candidates_items:
        if product_better(score, best_score):
            # Remove previous best candidate and add the new one
            del best_candidates[best_score]
            best_candidates[score] = [candidate]
            is_uncomp = False  # The candidates are comparable
            updated_keys = True
        if product_better(best_score, score):
            is_uncomp = False  # The candidates are comparable

    if is_uncomp:  # The two candidates are uncomparable
        best_candidates[score].append(candidate)
        best_candidates[score] = sorted(best_candidates[score], key=lambda x: len(x[0]) + len(x[1]))

        if len(best_candidates[score]) > num_params:
            best_candidates[score] = [cand for cand in best_candidates[score] if
            (len(cand[0]) + len(cand[1])) <= len(best_candidates[score][0][0]) + len(best_candidates[score][0][1]) + 2]
            # best_candidates[score] = best_candidates[score][:num_params]

    return updated_keys

def generate_asexual_part(best_candidates, temp, part, generate_func, all_colors, config, alpha_mutate_rule_same_type):
    if type(part) == list:
        if np.random.rand() < (1 / (len(part) + 1))**0.75:
            part.append(generate_func(all_colors, best_candidates, temp, config))
        else:
            index = np.random.randint(len(part))
            if np.random.rand() < 0.3:
                part = part[:index] + part[(index + 1):]
            else:
                r_type = None
                if np.random.rand() < alpha_mutate_rule_same_type:
                    r_type = part[index]['type']
                if np.random.rand() < 0.5:
                    part[index] = generate_func(all_colors, best_candidates, temp, config, r_type)
                else:
                    part = part[:index] + [generate_func(all_colors, best_candidates, temp, config, r_type)] + part[index:]
    else:
        part = generate_func(all_colors, best_candidates, temp, config)
    return part


def generate_sexual_part(best_candidates, temp, first, second, generate_func, all_colors, config, alpha_sexual_mutate,
                         alpha_mutate_rule_same_type, alpha_mutate_rule_same_type_one_parameter):
    if type(first) == list:
        if len(first) == 0 and len(second) == 0:
            child = []

        elif len(first) == 0:
            split2 = np.random.randint(len(second))

            if np.random.rand() <= 0.5:
                child = second[split2:]
            else:
                child = second[:split2]

        elif len(second) == 0:
            split1 = np.random.randint(len(first))

            if np.random.rand() <= 0.5:
                child = first[split1:]
            else:
                child = first[:split1]

        else:
            split1 = np.random.randint(len(first))
            split2 = np.random.randint(len(second))

            if np.random.rand() <= 0.5:
                child = first[:split1] + second[split2:]
            else:
                child = second[:split2] + first[split1:]

        if np.random.rand() < alpha_sexual_mutate:
            index = np.random.randint(len(child) + 1)
            if index == len(child):
                child.append(generate_func(all_colors, best_candidates, temp, config))
            else:
                r_type = None
                same_type = np.random.rand() < alpha_mutate_rule_same_type
                one_param_modification = np.random.rand() < alpha_mutate_rule_same_type_one_parameter
                if same_type:
                    r_type = child[index]['type']
                    same_type_rule = generate_func(all_colors, best_candidates, temp, config, r_type)
                    if not one_param_modification:
                        child[index] = same_type_rule
                    else:
                        key = random.choice(list(child[index].keys()))
                        child[index][key] = same_type_rule[key]
                else:
                    if np.random.rand() < 0.5:
                        child[index] = generate_func(all_colors, best_candidates, temp, config)
                    else:
                        child = child[:index] + [generate_func(all_colors, best_candidates, temp, config, r_type)] + child[
                                                                                                                     index:]
    else:
        if np.random.rand() < 0.5:
            child = copy.deepcopy(first)
        else:
            child = copy.deepcopy(second)
    return child


def generate_asexual_child(best_candidates, temp, parent, all_colors, config, alpha_mutate_rule_same_type):
    child = copy.deepcopy(parent)

    gen_functions = [get_random_global_rule, get_random_ca_rule, get_random_split_rule, get_random_merge_rule]
    idx_to_mutate = np.random.choice(len(child), p =[0.4, 0.4, 0.1, 0.1])
    child[idx_to_mutate] = generate_asexual_part(best_candidates, temp, child[idx_to_mutate], gen_functions[idx_to_mutate],
                                                 all_colors, config, alpha_mutate_rule_same_type)
    return child


def generate_sexual_child(best_candidates, temp, first, second, all_colors, config, alpha_sexual_mutate,
                          alpha_mutate_rule_same_type, alpha_mutate_rule_same_type_one_parameter):

    gen_functions = [get_random_global_rule, get_random_ca_rule, get_random_split_rule, get_random_merge_rule]
    what_to_mutate = np.random.choice(len(gen_functions), p=[0.5, 0.5, 0.0, 0.0])

    child = []
    for idx_to_mutate, gen_func in enumerate(gen_functions):
        child.append(generate_sexual_part(best_candidates, temp, first[idx_to_mutate], second[idx_to_mutate],
                                          gen_func, all_colors, config,
                                          (what_to_mutate==idx_to_mutate) * alpha_sexual_mutate, alpha_mutate_rule_same_type,
                                    alpha_mutate_rule_same_type_one_parameter))

    return child

def post_solved_process(task, solved, all_colors, config, reverse_functions, config_mapping):
    test_preds = []

    best_candidates = defaultdict(list)
    update_pool(task, best_candidates, solved, 1)

    start_time = time.time()

    while time.time() - start_time < 30:
        best_scores = list(best_candidates.keys())
        first_score = random.choice(best_scores)
        idx = np.random.choice(len(list(best_candidates[first_score])))
        first = list(best_candidates[first_score])[idx]
        child = generate_asexual_child(best_candidates, 0.5, first, all_colors, config, 0.)
        update_pool(task, best_candidates, child, 1)

    train_colors_features, scaler, _ = get_train_color_features(task)
    print(list(best_candidates.values())[0][0])
    for i_task, uni_task in enumerate(task['test']):
        predictions = []
        for solved in list(best_candidates.values())[0]:
            if reverse_functions[i_task].get('train_colors_in', None):
                inp = uni_task['input']
                colors_unique, color_numbers = np.unique(inp, return_counts=True)

                input_grid = uni_task['input']
                colors_features, colors = get_color_features(input_grid)
                colors_features = scaler.transform(colors_features)

                colors_in = []
                colors_out = []
                if colors_unique.shape[0] <= train_colors_features.shape[0]:
                    proper_colors = list(np.arange(train_colors_features.shape[0]))
                    for i, color in enumerate(colors):
                        color_features = colors_features[i].reshape((1, -1))
                        distances = np.sum(np.power(train_colors_features - color_features, 2), axis=1)
                        closests = list(np.argsort(distances))
                        for closest in closests:
                            if closest in proper_colors:
                                proper_colors.remove(closest)
                                colors_in.append(color)
                                colors_out.append(closest)
                                break

                    colors_in += list(reverse_functions[i_task]['train_colors_out'])
                    colors_out += list(reverse_functions[i_task]['train_colors_in'])

                    input_task = change_color(colors_in, colors_out, uni_task['input'])

                    trace = trace_param_automata(input_task, solved, 25, 0)
                    t_pred = trace[-1][0]

                    if not reverse_functions[i_task].get('unreduce_grid', None) is None:
                        t_pred = reverse_functions[i_task]['unreduce_grid'](t_pred)
                    if not reverse_functions[i_task].get('train_colors_in', None) is None:
                        colors_in = reverse_functions[i_task]['train_colors_in'] + reverse_functions[i_task][
                            'test_colors_in']
                        colors_out = reverse_functions[i_task]['train_colors_out'] + reverse_functions[i_task][
                            'test_colors_out']
                        t_pred = change_color(colors_in, colors_out, t_pred)
                    predictions.append(t_pred)
                else:
                    closests_to = [[] for _ in range(train_colors_features.shape[0])]
                    for i, color in enumerate(colors):
                        color_features = colors_features[i].reshape((1, -1))
                        distances = np.sum(np.power(train_colors_features - color_features, 2), axis=1)
                        closest = np.argsort(distances)[0]
                        closests_to[closest].append(color)

                    for i in range(len(closests_to)):
                        if len(closests_to[i]) == 0:
                            closests_to[i] = [-1]

                    answers = []
                    for color_map in product(*closests_to):
                        input_task = np.zeros_like(uni_task['input'])

                        for i, color in enumerate(list(color_map)):
                            input_task[uni_task['input'] == color] = i

                        colors_in = np.array(list(color_map) + reverse_functions[i_task]['test_colors_out'])
                        colors_out = list(np.arange(colors_in.shape[0])) + reverse_functions[i_task]['test_colors_in']

                        trace = trace_param_automata(input_task, solved, 25, 0)
                        t_pred = trace[-1][0]
                        t_pred = change_color(colors_out, colors_in, t_pred)
                        if not reverse_functions[i_task].get('unreduce_grid', None) is None:
                            t_pred = reverse_functions[i_task]['unreduce_grid'](t_pred)

                        answers.append(t_pred)

                    shapes = [ans.shape for ans in answers]
                    diff_shapes, counts = np.unique(shapes, return_counts=True, axis=0)
                    best_shape = diff_shapes[np.argmax(counts)]
                    answers = [ans for ans in answers if ans.shape == tuple(best_shape)]
                    final_answer = np.zeros((10, best_shape[0], best_shape[1]))
                    for i in range(10):
                        for ans in answers:
                            final_answer[i][ans == i] += 1
                    final_answer = np.argmax(final_answer, axis=0)

                    predictions.append(final_answer)

            else:
                inp = uni_task['input']

                trace = trace_param_automata(inp, solved, 25, 0)
                t_pred = trace[-1][0]

                if not reverse_functions[i_task].get('unreduce_grid', None) is None:
                    t_pred = reverse_functions[i_task]['unreduce_grid'](t_pred)

                if not reverse_functions[i_task].get('return_wall', None) is None:
                    t_pred = reverse_functions[i_task]['return_wall'](t_pred)

                predictions.append(t_pred)


        shapes = [ans.shape for ans in predictions]
        diff_shapes, counts = np.unique(shapes, return_counts=True, axis=0)
        best_shape = diff_shapes[np.argmax(counts)]
        predictions = [ans for ans in predictions if ans.shape == tuple(best_shape)]

        unique_preds, nums = np.unique(np.array(predictions), return_counts=True, axis=0)

        indexes = np.argsort(nums)[::-1]

        preds = unique_preds[indexes[:3]]
        preds = [pr for pr in preds]
        test_preds.append(preds)

    return test_preds


def train_model(name, task, params, time_for_task, config_mapping):
    alpha_asexual_mutation = params['alpha_asexual_mutation']
    alpha_sexual_mutate = params['alpha_sexual_mutate']
    alpha_mutate_rule_same_type = params['alpha_mutate_rule_same_type']
    alpha_mutate_rule_same_type_one_parameter = params['alpha_mutate_rule_same_type_one_parameter']
    add_random = params['add_random']
    num_params = params['num_params']
    start_time = time.time()
    param_name = str([alpha_asexual_mutation,
                      alpha_sexual_mutate,
                      alpha_mutate_rule_same_type,
                      alpha_mutate_rule_same_type_one_parameter,
                      add_random])

    task, reverse_functions = build_mapping(task, config_mapping)

    all_colors, config = get_task_metadata(task)

    print(f'Trying to solve {name}... {param_name}')

    best_candidates = defaultdict(list)
    test_preds = []
    population = generate_population(all_colors, config, size=2500)

    mode = 'test'
    # #
    # cand = [[{'type': 'flip', 'macro_type': 'global_rule', 'apply_to': 'index', 'apply_to_index': 5, 'how': 'hor'}],
    #         [], {'type': 'macro_multiply', 'k': (3, 3)}, {'type': 'cellwise_or'}]
    # #
    #update_pool(task, best_candidates, cand, num_params)
    # 1/0
    for cand in population:
        update_pool(task, best_candidates, cand, num_params)

    # print('Population generated')
    i_iteration = 0
    updated = 0

    num_successful_asexuals = 0
    num_asexuals = 0
    num_successful_sexuals = 0
    num_sexuals = 0
    while True:
        was_asexual = False
        was_sexual = False
        temp = min(0.9, (time.time() - start_time) / 500)

        if np.random.rand() < add_random:
            split_rule = get_random_split_rule(all_colors, {}, 0, config)
            merge_rule = get_random_merge_rule(all_colors, {}, 0, config)
            child = [generate_random_global(all_colors, best_candidates, temp, config),
                     generate_random_ca(all_colors, best_candidates, temp, config), split_rule, merge_rule]

        else:
            best_scores = list(best_candidates.keys())
            first_score = random.choice(best_scores)
            first = random.choice(list(best_candidates[first_score]))
            if np.random.rand() < alpha_asexual_mutation:
                child = generate_asexual_child(best_candidates, temp, first, all_colors, config,
                                               alpha_mutate_rule_same_type)
                was_asexual = True
            else:

                second_score = random.choice(best_scores)
                second = random.choice(list(best_candidates[second_score]))

                child = generate_sexual_child(best_candidates, temp, first, second, all_colors, config,
                                              alpha_sexual_mutate,
                                              alpha_mutate_rule_same_type,
                                              alpha_mutate_rule_same_type_one_parameter)
                was_sexual = True
        #print(was_asexual, was_sexual)
        #print(child)
        updated_keys = update_pool(task, best_candidates, child, num_params)

        if was_asexual:
            num_asexuals += 1
            if updated_keys:
                num_successful_asexuals += 1
        elif was_sexual:
            num_sexuals += 1
            if updated_keys:
                num_successful_sexuals += 1

        if i_iteration % 100 == 0:
            solved = None
            max_scores = np.zeros(len(list(best_candidates.keys())[0]))
            for score, params in best_candidates.items():
                max_scores = np.maximum(max_scores, score)
                if np.mean(score) == 1.:
                    if mode == 'test':
                        solved = params[0]
                        break

            # print(np.round(max_scores, 4), len(list(best_candidates.keys())), np.round(temp, 3),
            #      num_successful_sexuals, num_sexuals, num_successful_asexuals, num_asexuals)

            # if i_iteration % 1000==0:
            #     max_ps = None
            #     max_acc = 0
            #     for score, params in best_candidates.items():
            #         if max_acc < score[0]:
            #             max_acc = max(max_acc, score[0])
            #             max_ps = params[0]
            #     print(max_ps)

            if solved is not None:
                break

        if time.time() - start_time >= time_for_task:
            break
        i_iteration += 1

    if solved is not None:
        print(f'Solved {name}', time.time() - start_time)
        # vis_automata_paramed_task(task, solved, 25, 0)
        test_preds = post_solved_process(task, solved, all_colors, config, reverse_functions,
                                         config_mapping)
        #print(test_preds)


    else:
        for task_test in task['test']:
            test_preds.append([np.zeros_like(task_test['input'])])
    return solved, test_preds


if False:
    task_ids = train_task_ids
    mode = 'training'
elif False:
    task_ids = valid_task_ids
    mode = 'evaluation'
else:
    task_ids = test_task_ids
    mode = 'test'

good_tasks_number = 0
all_tasks = []

for task_id in range(len(task_ids)):
    name = task_ids[task_id]
    task = load_data(f'{name}.json', phase=mode)
    all_tasks.append(task_id)
    good_tasks_number += 1


good_tasks = np.arange(100) # [22,23,25,26,27,28,29]

time_for_task = 400

num_params = 5
alpha_asexual_mutation = 0.5
alpha_sexual_mutate = 0.5
alpha_mutate_rule_same_type = 0.5
alpha_mutate_rule_same_type_one_parameter = 0.0
alpha_add_random = 0.05
test_predictions = []


def try_to_solve_task(task_id_big):
    if task_id_big >= 100:
        config_mapping = {'map_color': True, 'reduce_grid': True, 'find_wall': False}
    else:
        config_mapping = {'map_color': False, 'reduce_grid': True, 'find_wall': True}

    task_id = task_id_big % 100

    name = task_ids[task_id]
    task = load_data(f'{name}.json', phase=mode)

    exclude_list = []
    is_public_test = ('00576224' in test_task_ids)
    if not task_id in good_tasks or (task_id in exclude_list) or (is_public_test and task_id > 3):
        test_preds = []
        if task_id in exclude_list:
            for task_test in task['test']:
                test_preds.append([np.ones_like(task_test['input'])])
        else:
            for task_test in task['test']:
                test_preds.append([np.zeros_like(task_test['input'])])

        return (test_preds, False), task_id_big

    params = {}
    params['alpha_asexual_mutation'] = alpha_asexual_mutation
    params['alpha_sexual_mutate'] = alpha_sexual_mutate
    params['alpha_mutate_rule_same_type'] = alpha_mutate_rule_same_type
    params['alpha_mutate_rule_same_type_one_parameter'] = alpha_mutate_rule_same_type_one_parameter
    params['add_random'] = alpha_add_random
    params['num_params'] = num_params
    solved, test_preds = train_model(name, task, params, time_for_task, config_mapping)

    return (test_preds, solved is not None), task_id_big


def main():
    answers = {}
    solved_tasks = 0

    if ENABLE_MULTIPROCESSING:
        pool = multiprocessing.Pool()
        iterator = pool.imap_unordered(try_to_solve_task, range(2*len(task_ids)))
    else:
        iterator = map(try_to_solve_task, range(2*len(task_ids)))


    for (test_preds, solved), answer_id in iterator:
        answers[answer_id] = test_preds

        if solved:
            solved_tasks += 1
            print(answer_id)
            print(test_preds)

    if ENABLE_MULTIPROCESSING:
        pool.close()
        pool.join()

    return answers, solved_tasks

answers, solved_tasks = main()

map_color_answers = {k:answers[k] for k in answers if k<100}

find_wall_answers = {k:answers[k] for k in answers if k>=100}

test_map_color_predictions = []
for answer_id in sorted(map_color_answers.keys()):
    test_map_color_predictions += map_color_answers[answer_id]

test_find_wall_predictions = []
for answer_id in sorted(find_wall_answers.keys()):
    test_find_wall_predictions += find_wall_answers[answer_id]


def flattener(preds):
    preds = copy.deepcopy(preds)
    str_final = ''
    str_preds = []
    for pred in preds:

        str_pred = str([row for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        str_preds.append(str_pred)
    return " ".join(str_preds)


test_map_color_predictions = [[[list(pred) for pred in test_pred] for test_pred in test_task] for test_task in test_map_color_predictions]
test_find_wall_predictions = [[[list(pred) for pred in test_pred] for test_pred in test_task] for test_task in test_find_wall_predictions]


submission = pd.read_csv(data_path / "sample_submission.csv")

for idx in range(len(test_map_color_predictions)):
    map_color_preds = test_map_color_predictions[idx]
    find_wall_preds = test_find_wall_predictions[idx]

    if np.mean(map_color_preds) == 0 and np.mean(find_wall_preds) == 0:
        submission.loc[idx, "output"] = "|0|"
    elif np.mean(map_color_preds) > 0 and np.mean(find_wall_preds) == 0:
        submission.loc[idx, "output"] = flattener(map_color_preds)
    elif np.mean(map_color_preds) == 0 and np.mean(find_wall_preds) > 0:
        submission.loc[idx, "output"] = flattener(find_wall_preds)
    else:
        preds_here = find_wall_preds + map_color_preds

        preds_here = copy.deepcopy(preds_here)
        str_final = ''
        str_preds = []
        for pred in preds_here:
            str_pred = str([row for row in pred])
            str_pred = str_pred.replace(', ', '')
            str_pred = str_pred.replace('[[', '|')
            str_pred = str_pred.replace('][', '|')
            str_pred = str_pred.replace(']]', '|')
            str_preds.append(str_pred)

        unique_preds, nums = np.unique(str_preds, return_counts=True)

        indexes = np.argsort(nums)[::-1]

        preds = unique_preds[indexes[:3]]
        preds = [pr for pr in preds]

        submission.loc[idx, "output"] = " ".join(preds)

print(submission.head())

submission.to_csv("submission.csv", index=False)

print(solved_tasks)
