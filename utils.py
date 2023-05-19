def add_activation_to_dict(name, dict_):
    def hook_fn(module, input, output):
        dict_[name] = output.detach().cpu()
    return hook_fn


def register_hooks(model, dict_):
    for name, module in model.named_modules():
        module.register_forward_hook(add_activation_to_dict(name, dict_))
                                     
