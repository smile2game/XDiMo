import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte import Latte_models
from .latte_img import LatteIMG_models
from .latte_t2v import LatteT2V
from .xdimo import xdimo_models

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'LatteIMG' in args.model:
        return LatteIMG_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    elif 'LatteT2V' in args.model:
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    elif 'xdimo' in args.model.lower():
        kwargs = dict(
            input_size=args.latent_size,
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            learn_sigma=args.learn_sigma,
            extras=args.extras,
        )
        if 'MoE' in args.model:
            if getattr(args, 'num_experts', None) is not None:
                kwargs['num_experts'] = args.num_experts
            if getattr(args, 'top_k', None) is not None:
                kwargs['top_k'] = args.top_k
            if getattr(args, 'expert_capacity_factor', None) is not None:
                kwargs['expert_capacity_factor'] = args.expert_capacity_factor
        return xdimo_models[args.model](**kwargs)
    elif 'Latte' in args.model:
        kwargs = dict(
            input_size=args.latent_size,
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            learn_sigma=args.learn_sigma,
            extras=args.extras,
        )
        if 'MoE' in args.model:
            if getattr(args, 'num_experts', None) is not None:
                kwargs['num_experts'] = args.num_experts
            if getattr(args, 'top_k', None) is not None:
                kwargs['top_k'] = args.top_k
            if getattr(args, 'expert_capacity_factor', None) is not None:
                kwargs['expert_capacity_factor'] = args.expert_capacity_factor
        return Latte_models[args.model](**kwargs)
    else:
        raise '{} Model Not Supported!'.format(args.model)
    