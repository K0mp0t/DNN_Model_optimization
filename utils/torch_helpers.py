from itertools import groupby
import time
import torch
from torch import nn
from tqdm import tqdm
from torchmetrics.text import CharErrorRate
from utils.data import decode_texts
from copy import deepcopy


def train_model(model, alphabet, epochs, train_loader, val_loader=None, lr=1e-3, early_stopping_patiece=10, device='cpu'):
    model.to(device)
    model.train()

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    criterion = ctc_loss_log_differentiable_torch
    metric = CharErrorRate()

    train_loss_history = list()
    val_loss_history = list()

    train_cer_history = list()
    val_cer_history = list()

    best_model_state = None
    best_model_val_loss = torch.finfo(torch.float32).max

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        train_cer = 0
        val_cer = 0

        for i, ((x1, x2), y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x1, x2)

            input_lengths = torch.full((y_pred.shape[0],), y_pred.shape[1]).to(device)
            target_lengths = torch.sum(y != 0, axis=1)

            loss = criterion(torch.log(y_pred), y, input_lengths, target_lengths, device=device)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_cer += metric(decode_texts(y_pred.detach().cpu().numpy(), alphabet, blank_idx=0),
                                [''.join(alphabet[k - 1] for k, _ in groupby(e) if k != 0) for e in
                                 y.cpu().numpy().astype(int)]).item()

            print(f'\rEpoch {epoch}, {i + 1}/{len(train_loader)}, loss: {round(train_loss / (i + 1), 6)}, cer: {round(train_cer / (i + 1), 6)}',
                end='')
        if val_loader is not None:
            with torch.no_grad():
                for i, ((x1, x2), y) in enumerate(val_loader):
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y = y.to(device)

                    y_pred = model(x1, x2)

                    input_lengths = torch.full((y_pred.shape[0],), y_pred.shape[1]).to(device)
                    target_lengths = torch.sum(y != 0, axis=1)
                    loss = criterion(torch.log(y_pred), y, input_lengths, target_lengths, device=device)

                    val_loss += loss.item()
                    val_cer += metric(decode_texts(y_pred.detach().cpu().numpy(), alphabet, blank_idx=0),
                                      [''.join(alphabet[k - 1] for k, _ in groupby(e) if k != 0) for e in
                                       y.cpu().numpy().astype(int)]).item()
            print(f', val_loss: {round(val_loss / len(val_loader), 6)}, val_cer: {round(val_cer / len(val_loader), 6)}')
            val_loss_history.append(val_loss / len(val_loader))
            val_cer_history.append(val_cer / len(val_loader))
            lr_scheduler.step(val_loss / len(val_loader))
        else:
            print()

        train_loss_history.append(train_loss / len(train_loader))

        train_cer_history.append(train_cer / len(train_loader))

        if val_loader is not None and val_loss_history[-1] < best_model_val_loss:
            best_model_val_loss = val_loss_history[-1]
            best_model_state = deepcopy(model.state_dict())

        if val_loader is not None and min(val_loss_history) < min(val_loss_history[-early_stopping_patiece:]):
            print('Early stopping')
            break

    best_model_state = model.state_dict() if best_model_state is None else best_model_state
    history = {'train_loss': train_loss_history, 'train_cer': train_cer_history,
               'val_loss': val_loss_history, 'val_cer': val_cer_history}
    return best_model_state, history


def validate_model(model, dataloader, alphabet, runs=10, device='cpu'):
    model.to(device)
    model.eval()

    criterion = nn.CTCLoss()
    metric = CharErrorRate()

    cumtime = 0

    with torch.no_grad():
        for j in range(runs):
            loss = 0
            cer_value = 0
            for i, ((x1, x2), y) in enumerate(dataloader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                start = time.time()
                y_pred = model(x1, x2)
                cumtime += time.time() - start

                input_lengths = torch.full((y_pred.shape[0],), y_pred.shape[1]).to(device)
                target_lengths = torch.sum(y != 0, dim=1)
                loss += criterion(torch.log_softmax(y_pred, -1).permute(1, 0, 2), y, input_lengths, target_lengths).item()
                cer_value += metric(decode_texts(y_pred.detach().cpu().numpy(), alphabet, blank_idx=0),
                                    [''.join(alphabet[k-1] for k, _ in groupby(e) if k != 0)
                                    for e in y.cpu().numpy().astype(int)]).item()

    return cumtime / len(dataloader) / runs, loss / len(dataloader), cer_value / len(dataloader)


@torch.no_grad()
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def ctc_loss_log_differentiable_torch(log_logits: torch.FloatTensor, targets: torch.LongTensor,
                                      input_lengths: torch.Tensor, target_lengths: torch.Tensor, device,
                                      blank_idx=0, dtype_to_use=torch.float32) -> torch.float32:
    """
    log_logits: np.ndarray of shape (B, T, C)
    targets: np.ndarray of shape (B, L,)
    """

    B, T = log_logits.shape[0], log_logits.shape[1]
    S = 2 * targets.shape[1] + 1

    zero = torch.finfo(dtype_to_use).min

    # insert blanks between every pair of labels and add them to start and end of the seq
    extended_targets = torch.stack([torch.full_like(targets, blank_idx), targets], dim=-1).flatten(start_dim=-2)
    extended_targets = torch.cat([extended_targets, torch.full((B, 1), blank_idx, device=device)], dim=-1)
    # due to the paper formula for alpha_t(s) we must know where labels repeat and where the blanks are
    # in the extended label seq
    targets_difference_mask = torch.cat([torch.full((B, 2), False, device=device),
                                         extended_targets[:, 2:] != extended_targets[:, :-2]], dim=-1)

    # initialize alphas array to keep track of previous alphas
    # (also add 2 to the second dim so our s-2 and s-1 vectorized calculations won't get IndexError)
    log_alphas = torch.full((B, T, S+2), zero, dtype=dtype_to_use, device=device)

    # every accountable prefix starts either with a blank or the first symbol of the target,
    # so we initialize alphas in the following way (remember about S+2)
    log_alphas[:, 0, 2] = log_logits[:, 0, blank_idx]
    log_alphas[:, 0, 3] = log_logits[torch.arange(B), 0, targets[:, 0]]

    for t in range(1, T):
        # remember we're in log space so log(a*b) = log(a) + log(b)
        # here formula must be mathematically reworked.

        log_alphas[:, t, 2:] = (torch.gather(log_logits[:, t], -1, extended_targets) +
                                torch.logsumexp(torch.stack([log_alphas[:, t-1, 2:], log_alphas[:, t-1, 1: -1],
                                                             torch.where(targets_difference_mask,
                                                                         log_alphas[:, t-1, :-2], zero)]), dim=0))

    temp = torch.gather(log_alphas[torch.arange(B), input_lengths-1], -1,
                        torch.stack([2 + target_lengths * 2 - 1, 2 + target_lengths * 2], dim=-1))

    return -torch.mean(torch.logsumexp(temp, dim=-1))


def warmup_torch_model(model, input_shapes, device='cpu'):
    for i in range(10):
        model(*(torch.randn(s).to(device) for s in input_shapes))