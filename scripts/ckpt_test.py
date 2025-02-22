import torch
import argparse

def view_checkpoint(checkpoint_path):
    """
    View the basic information of the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    print("Basic Information in Checkpoint:")
    print(f"- Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    print("\nKeys in Checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    print("\nKeys in Model State Dict:")
    model_state_dict = checkpoint.get("model_state_dict", {})
    for key, value in model_state_dict.items():
        print(f"  - {key}, shape is:{value.shape}")
        # if "_attn_mask" in key:
        #     print(f"Found: {key}")
        #     print(f"Shape: {value.shape}")
        #     print(f"Values: {value}")
    
    print("\nKeys in Optimizer State Dict:")
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", {})
    for key in optimizer_state_dict.keys():
        print(f"  - {key}")
    # print("\nKeys in the param_groups:")
    # for key in optimizer_state_dict["param_groups"]:
    #     print(f"  - {key}")
    # print("\nKeys in the state:")
    # for key in optimizer_state_dict["state"]:
    #     print(f"  - {key}")


def view_checkpoint_details(checkpoint_path):
    """
    View detailed information of the checkpoint, including parameter shapes.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    print("Detailed Information in Checkpoint:")
    print(f"- Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    print("\nModel Parameters:")
    model_state_dict = checkpoint.get("model_state_dict", {})
    for key, value in model_state_dict.items():
        print(f"  - {key}: {value.shape}")
    
    print("\nOptimizer State:")
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", {})
    print("  - State keys:")
    for key, value in optimizer_state_dict.get("state", {}).items():
        print(f"    - {key}: {value.keys()}")
    print("  - Param groups:")
    for group in optimizer_state_dict.get("param_groups", []):
        print(f"    - {group}")

ckpt_path="/home/yinj@/datas/grkvc/ckpts/ml-20m/model_1.ckpt"
default_ckpt_path="/home/yinj@/datas/grkvc/ckpts/ml-20m/model_base.ckpt"

ckpt_path=default_ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View checkpoint information.")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        nargs="?",
        default=ckpt_path,
        help="Path to the checkpoint file (default: checkpoint.pth)."
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="View detailed information about the checkpoint."
    )
    
    args = parser.parse_args()
    
    if args.details:
        view_checkpoint_details(args.checkpoint_path)
    else:
        view_checkpoint(args.checkpoint_path)
