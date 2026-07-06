#
#   Get max batch size from network in current GPU
#
#   DGPP - Nov-2025
#

import torch
import torchvision.models as tvm
import csv

# # https://github.com/rwightman/pytorch-image-models
from timm import create_model


# ---------------------------------------------------------
#           CREATE MODELS FROM TIMM or
# ---------------------------------------------------------
def create_model_timm(name):
    try:
        return create_model(name.lower(), pretrained=False)
    except Exception as e:
        raise ValueError(f"Error while creating model '{name}' from TIMM: {e}")

def create_model_pytorch(name):
    name = name.lower()
    try:
        if not hasattr(tvm, name):
            raise ValueError(f"Model '{name}' doesn't exist in torchvision.models")

        return getattr(tvm, name)(weights=None)
    except Exception as e:
        raise ValueError(f"Error while creating model '{name}' from PyTorch: {e}")


MODEL_REGISTRY = {}  # para custom

def create_model_(name, source="auto", pretrained=False, **kwargs):
    name_l = name.lower()

    try:
        # 1) Tenta TIMM
        if source in ("auto", "timm"):
            try:
                return create_model(name_l, pretrained=pretrained, **kwargs)
            except:
                if source == "timm":
                    raise

        # 2) Tenta torchvision
        if source in ("auto", "torchvision"):
            if hasattr(tvm, name_l):
                print('Using torchvision')
                return getattr(tvm, name_l)(weights=None, **kwargs)

        # 3) Tenta registry custom
        if source in ("auto", "custom"):
            if name_l in MODEL_REGISTRY:
                return MODEL_REGISTRY[name_l](**kwargs)

        raise ValueError(f"Model '{name}' not found anywhere.")

    except Exception as e:
        raise ValueError(f"Error while creating model '{name}': {e}")

# ---------------------------------------------------------
#       TESTE SE O BATCH CABE NA GPU
# ---------------------------------------------------------
def fits_in_memory(model_fn, batch_size, device, input_size):
    try:
        torch.cuda.empty_cache()

        model = model_fn().to(device)
        model.train()

        x = torch.randn(batch_size, *input_size, device=device)

        y = model(x)
        loss = y.sum()
        loss.backward()

        del model, x, y, loss
        torch.cuda.empty_cache()

        return True

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return False
        raise e


# ---------------------------------------------------------
#   GERADOR DE BATCH SIZE BASEADO NO MODO SELECIONADO
#   We should by default try to fit in power2 sizes
#   Starts from 2, as some models can't run with BS=1
# ---------------------------------------------------------
def batch_generator(max_test, mode="power2"):
    """
    mode="power2"     -> apenas potências de 2: 1,2,4,8,...
    mode="multiple2"  -> múltiplos de 2: 2,4,6,8,...
    mode="any"        -> 1,2,3,4,... (comportamento original)
    """

    if mode == "power2":
        b = 2   #TODO 2- dá problema com rede grande como convnext  #1- dá problema com mobilenetv4_conv_small
        while b <= max_test:
            yield b
            b *= 2

    elif mode == "multiple2":
        for b in range(2, max_test + 1, 2):
            yield b

    else:  # "any"
        for b in range(2, max_test + 1):    #iniciava em 1
            yield b


# ---------------------------------------------------------
#       FIND BIGGEST BATCH FOR CHOSEN MODEL
# ---------------------------------------------------------
def find_max_batch(model_fn, device, input_size, max_test, batch_mode):
    best = 0

    for batch_size in batch_generator(max_test, batch_mode):
        if fits_in_memory(model_fn, batch_size, device, input_size):
            best = batch_size
        else:
            break

    return best


# ---------------------------------------------------------
#         MAIN FUNCTION
# ---------------------------------------------------------
def find_max_bs(
    model_names,
    input_size=(3, 224, 224),
    csv_path="timm_max_batch.csv",
    batch_mode="power2",    # <- escolha aqui
    max_test=4096
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch mode: {batch_mode}\n")

    for name in model_names:
        print(f"== Testing {name} ==")

        model_fn = lambda name=name: create_model_(name)

        max_batch = find_max_batch(model_fn, device, input_size, max_test, batch_mode)
        results[name] = max_batch

        print(f"Maximum batch ({batch_mode}) for {name}: {max_batch}\n")

    # # Salvar CSV
    # with open(csv_path, "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Modelo TIMM", "Input Size", "Batch Mode", "Max Batch Size"])
    #     for name, max_batch in results.items():
    #         writer.writerow([name, str(input_size), batch_mode, max_batch])

    # print(f"Resultados salvos em: {csv_path}")
    return results


# ---------------------------------------------------------
#                 USAGE - 
#   We should by default try to fit in power2 sizes
# ---------------------------------------------------------
if __name__ == "__main__":
    models_timm = ['resnet18', 'resnet50', 'mobilenetv2_100', 'mobilenetv3_large_100', 'mobilenetv4_conv_small']

    max_bs = find_max_bs(
        models_timm,
        input_size=(3, 1152, 896),
        batch_mode="power2",     # "power2", "multiple2", "any"
        max_test=2048
    )
    print(max_bs)
    print(max_bs['resnet18'])