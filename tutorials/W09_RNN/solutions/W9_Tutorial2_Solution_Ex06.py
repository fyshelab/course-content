def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    # Truncate
    if len(line) > num_steps:
        return line[:num_steps]

    # Pad
    number_of_pad_tokens = num_steps - len(line)
    padding_list = [padding_token] * number_of_pad_tokens

    return line + padding_list
