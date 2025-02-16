@torch.no_grad()
def lring_prefill_and_decode(model, input_ids, position_ids, context_len, query_ids = None, max_new_tokens = 12, stop_ids = [], debug=False):
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    if query_ids is None:
        query_ids = input_ids[:, context_len:]
    output = model(input_ids[:, :context_len], query_ids=query_ids, position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug)
    
    past_key_values = output.past_key_values
    output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)
    
    if rank == world_size - 1:
        past_key_values = output.past_key_values
    hiddens = []
    if debug and rank == world_size - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if rank == world_size - 1: # generate next_id on the last gpu
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and rank == world_size - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if rank == world_size - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
        dist.broadcast(next_id, src=world_size-1)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids