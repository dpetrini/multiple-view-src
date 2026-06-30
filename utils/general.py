# Convert channels from 3 to 1 for all images in directories below
import os
import cv2
import numpy as np
import torch
from datetime import datetime
import pytz

# assign directory
directory = './'
 
# iterate and save over files in that directory
img_list = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.png'):
            img_list.append(os.path.join(root, filename))

print('Will try to convert ', len(img_list), ' images.')

# img_list = [i for i in os.listdir()]
count = 0
for i, img in enumerate(img_list):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        continue
    new_image = np.zeros((*image.shape[0:2], 1), dtype=np.uint16)
    new_image = image[:, :, 0]   # convert to one channel
    cv2.imwrite(img, new_image)
    print('.', end='', flush=True)
    count += 1
print('Converted ', count, ' images.')


def print_finish_time():
    # Get the current time in UTC
    utc_now = datetime.now(pytz.utc)

    # Convert UTC time to GMT-3 timezone as
    gmt_minus_three = pytz.timezone('America/Sao_Paulo')
    gmt_minus_three_time = utc_now.astimezone(gmt_minus_three)

    # Print the current date and time in GMT-3 timezone
    return "[End of training] Current date and time in GMT-3:", gmt_minus_three_time.strftime('%Y-%m-%d %H:%M:%S')


# helper functions
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def get_samples_n(dataset):
    # To calculate AUC error:
    if dataset == 'CBIS-DDSM':
        m=264
        n=381
    elif 'SMALL' in dataset:      # Vindr SMALL
        m=20
        n=20
    elif dataset in ['VINDR_BIRADS', 'VINDR_MAMMO']:
        m=384
        n=3616
    return m, n

# print model description and layers
# not tested yet, maybe import torch for model.
def print_layers_names(model):
    print(model)
    cont = 0
    for name, param in model.named_parameters():
        print(cont, name)
        cont+=1

def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    #trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    #return f'{total/1e6:.2f} M'
    # return f'{total/1e6:,}'.replace(",", ".")
    return f'{total:,}'.replace(",", ".")


# Helper functions to process

# Retrieve the index of highest validation and clear tensor information from values
def get_best_val(number_runs, single_val_results, model_name):
    val_auc, high_val_auc  = 0, 0
    for i in range(number_runs):
        # Clean tensor information from training
        for name in single_val_results[model_name][i]['metric_name']:
            metric_name = name
            if metric_name == 'Accuracy':
                # print(single_val_results[model_name][i]['best_metric'][metric_name])
                single_val_results[model_name][i]['best_metric'][metric_name] = float(single_val_results[model_name][i]['best_metric'][metric_name])
                print(i, '---->>>> val_ACC ', single_val_results[model_name][i]['best_metric'][metric_name])

            if metric_name == 'AUC':
                # We only want metric_name = 'AUC'
                cur_val = single_val_results[model_name][i]['best_metric']['AUC']
                print(i, '---->>>> val_AUC ', cur_val)
                if  cur_val > val_auc:
                    val_auc = cur_val
                    high_val_auc = i
    return high_val_auc

# Calculate mean of test runs
def get_test_mean(number_runs, single_test_results, model_name):
    main, std = [], []
    for i in range(number_runs):
        content = single_test_results[model_name][i][0]
        # print(content)
        main.append(float(content[:6]))
        std.append(float(content[7:]))
    main = torch.tensor(main)
    std = torch.tensor(std)
    test_mean = f'{torch.mean(main):1.4f}±{torch.mean(std):1.4f}'
    # high_test_auc = torch.argmax(main)
    return test_mean

# Retrive value from key-name from a list with dict items
def get_instance(list_item, item_name): 
    result = None
    for item in list_item:
        # print('item: ', item)
        if isinstance(item, dict):
            if item_name in item:
                result = item[item_name]
    return result



# use in cmdline
def show_layers_size(model_file):

    checkpoint = torch.load(model_file, map_location='cpu')

    # Se for state_dict direto ou extrair de checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Calcula e exibe tamanho de cada camada
    total = 0
    accumulated = 0
    print(f"{'Camada':<60} {'Shape':<25} {'Params':>10} {'Acumulado':>12} {'MB':>8}")
    print("-" * 120)

    for name, tensor in state_dict.items():
        num_params = tensor.numel()
        size_mb = tensor.element_size() * num_params / (1024 ** 2)
        total += num_params
        accumulated += num_params
        print(f"{name:<60} {str(list(tensor.shape)):<25} {num_params:>10,} {accumulated:>12,} {size_mb:>8.4f}")

    print("-" * 120)

    total_mb = sum(
        t.element_size() * t.numel() / (1024**2)
        for t in state_dict.values()
    )
    print(f"{'TOTAL':<60} {'':<25} {total:>10,} {accumulated:>12,} {total_mb:>8.4f}")