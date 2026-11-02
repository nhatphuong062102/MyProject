import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model_clear import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


# ======================================================================================
# [MỚI] Thích ứng ViT với kích thước ảnh đầu vào khác 224 (đọc từ cfg.INPUT.SIZE)

# ======================================================================================


def resample_pos_embed(pos_embed: torch.Tensor, new_grid):
    """
    Nội suy bicubic positional embedding của ViT sang lưới patch mới.

    pos_embed : [1 + g*g, D]  (dòng 0 là token CLS, g*g dòng còn lại là patch; lưới gốc phải vuông)
    new_grid  : (new_gh, new_gw) = (H // patch_size, W // patch_size)
    return    : [1 + new_gh*new_gw, D]  (dòng CLS giữ nguyên, không nội suy)
    """
    n_patch = pos_embed.shape[0] - 1
    g = int(round(n_patch ** 0.5))
    assert g * g == n_patch, f"Positional embedding gốc không phải lưới vuông (n_patch={n_patch})"
    new_gh, new_gw = new_grid
    if (new_gh, new_gw) == (g, g):
        return pos_embed

    cls_pe, grid_pe = pos_embed[:1], pos_embed[1:]
    grid_pe = grid_pe.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()          # [1, D, g, g]
    grid_pe = F.interpolate(grid_pe, size=(new_gh, new_gw), mode="bicubic", align_corners=False)
    grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(new_gh * new_gw, -1).to(pos_embed.dtype)  # [new_gh*new_gw, D]
    return torch.cat([cls_pe, grid_pe], dim=0)


def adapt_visual_to_input_size(model, input_size):
    """
    Làm cho image encoder ViT của CLIP chạy được với ảnh kích thước input_size (đọc từ cfg.INPUT.SIZE).
    Chỉ thay positional embedding (nội suy từ bảng gốc 14x14 của ViT-B/16); mọi trọng số khác giữ nguyên.

    input_size : int hoặc (H, W); H và W phải chia hết cho patch_size.
    Với backbone không phải ViT (ví dụ RN50): không làm gì, chỉ cảnh báo.
    Nếu lưới patch mới trùng lưới gốc (ví dụ 224 với ViT-B/16): trả lại model nguyên vẹn.
    """
    visual = model.visual
    if not (hasattr(visual, "positional_embedding") and hasattr(visual, "conv1")):
        warnings.warn("adapt_visual_to_input_size: backbone không phải ViT, bỏ qua.")
        return model

    if isinstance(input_size, int):
        H = W = input_size
    else:
        H, W = int(input_size[0]), int(input_size[1])
    patch = visual.conv1.kernel_size[0]
    assert H % patch == 0 and W % patch == 0, (
        f"INPUT.SIZE=({H}, {W}) phải chia hết cho patch_size={patch}"
    )
    new_grid = (H // patch, W // patch)

    old_pe = visual.positional_embedding
    old_g = int(round((old_pe.shape[0] - 1) ** 0.5))
    if new_grid == (old_g, old_g):
        print(f"[adapt_visual_to_input_size] Lưới patch {old_g}x{old_g} trùng bảng gốc, giữ nguyên positional embedding.")
        return model

    new_pe = resample_pos_embed(old_pe.data, new_grid)
    visual.positional_embedding = nn.Parameter(new_pe, requires_grad=old_pe.requires_grad)
    visual.input_resolution = H if H == W else (H, W)
    print(f"[adapt_visual_to_input_size] Input {H}x{W}: positional embedding {tuple(old_pe.shape)} -> {tuple(new_pe.shape)} "
          f"(lưới patch {old_g}x{old_g} -> {new_grid[0]}x{new_grid[1]}, số local feature N = {new_grid[0] * new_grid[1]})")
    return model