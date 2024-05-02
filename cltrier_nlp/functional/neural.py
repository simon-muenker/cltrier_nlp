import torch


def get_device() -> str:
    """Return the computation device as a string based on the availability of CUDA.

    Returns:
        str: A PyTorch device object representing the appropriate device for computation.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def calculate_model_memory_usage(model: torch.nn.Module) -> str:
    """Calculate the memory usage of the input model in megabytes.

    Args:
        model (torch.nn.Module): The input model for which memory usage needs to be calculated.

    Returns:
        str: A string formatted to represent the size of the nn.Module in megabytes.
    """
    usage_in_byte: int = sum(
        [
            sum([param.nelement() * param.element_size() for param in model.parameters()]),
            sum([buf.nelement() * buf.element_size() for buf in model.buffers()]),
        ]
    )

    return f"{usage_in_byte / (1024.0 * 1024.0):2.4f} MB"
