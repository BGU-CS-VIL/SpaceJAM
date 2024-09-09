import torch
from pathlib import Path
from train_args import get_argparser, parse_and_verify_args
from spacejam.trainer import trainer_from_args


def init_config(parser):
    config = vars(parse_and_verify_args(parser))

    if config['run_name'] is None:
        config['run_name'] = str(Path(config["data_folder"]).name)

    return config


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({device_name})")

    parser = get_argparser()
     
    config = init_config(parser)
    config['device'] = device
    config['device_name'] = device_name

    trainer = trainer_from_args(config)
    trainer.train()
    
    results = trainer.eval(log_vis=True)
    
    print("=" * 160)
    dict_to_print = {k: v.item() for k, v in results.items() if isinstance(v, torch.Tensor)}
    sorted_scores = sorted(dict_to_print.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_scores:
        print(f"{k:160s}: {v*100:.4f}%")
    print("=" * 160)


if __name__ == "__main__":
    main()
    


