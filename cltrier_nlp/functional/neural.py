import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def calculate_model_memory_usage(model: torch.nn.Module) -> str:
    usage_in_byte: int = sum(
        [
            sum([param.nelement() * param.element_size() for param in model.parameters()]),
            sum([buf.nelement() * buf.element_size() for buf in model.buffers()]),
        ]
    )

    return f"{usage_in_byte / (1024.0 * 1024.0):2.4f} MB"
