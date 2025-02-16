import argparse
from time import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from expt.learning_cvrp.utils import *


def load_subproblem_data(path):
    data = np.load(path)
    xs, offsets, node_idxs, n_subp_nodes, dists, = data['xs'], data['offsets'], data['subp_node_idxs'], data[
        'n_subp_nodes'], data['dists']
    node_idxs = np.split(node_idxs, np.cumsum(n_subp_nodes[:-1]))
    return Namespace(
        N=len(n_subp_nodes), d_node=xs.shape[-1],
        xs=xs, offsets=offsets, node_idxs=pad_each(node_idxs), n_subp_nodes=n_subp_nodes, dists=dists
    )


def to_tensor(x, device='cpu'):
    if x is None: return None
    dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32 if np.issubdtype(x.dtype,
                                                                                                 np.floating) else None
    return torch.tensor(x, dtype=dtype, device=device)


def get_prepare_subproblem(d=None, rotate=False, flip=False, perturb_node=False):
    """
    Reshape training data to the correct format
    """

    def prep_batch(b):
        b.labels = b.get('dists', None)
        b.x = b.xs[(b.offsets.reshape(-1, 1) + b.node_idxs[:, :b.n_subp_nodes.max()]).astype(int)]
        b_t = Namespace((k, to_tensor(b[k])) for k in model_keys)
        b_t.x, _ = MDVRProblem.transform_features(b_t.x, None, rotate=rotate, flip=flip, perturb_node=perturb_node)
        return b_t

    model_keys = ['x', 'n_subp_nodes', 'labels']

    if d is None:  # Generation time
        return prep_batch
    else:  # Training and evaluation time
        return lambda idxs: prep_batch(
            Namespace(((k, d[k][idxs]) for k in ['offsets', 'node_idxs', 'n_subp_nodes', 'dists']), xs=d.xs))


# Load trained model
def restore(args, net, opt=None):
    if args.step is None:  # default = None
        models = list(args.model_save_dir.glob('*.pth'))
        if len(models) == 0:
            print('No model checkpoints found')
            return None
        step, load_path = max((int(p.stem), p) for p in models)  # Load the max step
    else:  # the most accurate step model
        step, load_path = args.step, args.model_save_dir / f'{args.step}.pth'
    ckpt = torch.load(load_path, map_location=args.device)
    net.load_state_dict(ckpt['net'])
    if opt is not None:
        opt.load_state_dict(ckpt['opt'])
    print(f'Loaded network{"" if opt is None else " and optimizer"} from {load_path}')
    return ckpt['step']


# Build up the network structure
class SubproblemNetwork(nn.Module):
    def __init__(self, args, d):
        super(SubproblemNetwork, self).__init__()
        self.args = args
        self.d_hidden = d_hidden = args.d_hidden
        self.register_buffer('mean_dist', torch.tensor(d.dists.mean()))
        self.fc = nn.Linear(d.d_node, d_hidden)
        layer = nn.TransformerEncoderLayer(d_model=d_hidden, nhead=args.transformer_heads, dim_feedforward=d_hidden * 4,
                                           dropout=args.dropout)
        self.layers = nn.TransformerEncoder(layer, num_layers=args.n_layers)
        self.fc_out = nn.Linear(d_hidden, 1)

    def forward(self, d):
        """
        Contents of d:
        x: shape (n_batch, max_n_subp_nodes, d_node)
        n_subp_nodes: shape (n_batch,)
        labels: shape (n_batch,)
        """
        args = self.args
        x, n_subp_nodes, labels = d.x, d.n_subp_nodes, d.get('labels')
        n_batch, max_n_subp_nodes, _ = x.shape

        mask = torch.arange(max_n_subp_nodes, device=x.device).expand(n_batch,
                                                                      max_n_subp_nodes) >= n_subp_nodes.unsqueeze(
            1)  # (n_batch, max_n_subp_nodes)
        x = self.fc(x)
        x = self.layers(x.transpose(0, 1),
                        src_key_padding_mask=mask)  # Transformer takes in (max_n_subp_nodes, n_batch, d_node)

        outs = self.fc_out(x).squeeze(-1)  # (max_n_subp_nodes, n_batch)
        outs[mask.T] = 0
        preds = outs.sum(dim=0) / n_subp_nodes + self.mean_dist  # (n_batch,)

        if labels is None:
            return preds  # Predicted distance
        if args.loss.startswith('MSE'):
            return F.mse_loss(preds, labels)
        if args.loss.startswith('MAE'):
            return F.l1_loss(preds, labels)
        return F.smooth_l1_loss(preds, labels, beta=1.0)


def train(args, d, d_eval, d_generate):
    start_time = time()
    writer = SummaryWriter(log_dir=args.model_save_dir, flush_secs=10)

    net = SubproblemNetwork(args, d).to(args.device)
    opt = Adam(net.parameters(), lr=args.lr)
    start_step = restore(args, net, opt=opt)

    scheduler = CosineAnnealingLR(opt, args.n_steps, last_epoch=-1 if start_step is None else start_step)
    start_step = start_step or 0

    prep = get_prepare_subproblem(d, rotate=args.augment_rotate, flip=args.augment_flip,
                                  perturb_node=args.augment_perturb_node, perturb_route=args.augment_perturb_route)

    def log(text, **kwargs):
        print(f'Step {step}: {text}', flush=True)
        [writer.add_scalar(k, v, global_step=step, walltime=time() - start_time) for k, v in kwargs.items()]

    for step in range(start_step, args.n_steps + 1):
        if step % args.n_step_save == 0 or step == args.n_steps:  # save model every 1000 steps
            ckpt = dict(step=step, net=net.state_dict(), opt=opt.state_dict())
            torch.save(ckpt, args.model_save_dir / f'{step}.pth')

        if step % args.n_step_eval == 0:  # evaluate model at evaluation step, default = 1000
            evaluate(args, d_eval, net, log)

        if step % args.n_step_generate == 0 and step > 0:  # generate predictions every n_step_generate, if not set to infinity, not implement
            generate(args, d_generate, net, step)

        if step == args.n_steps: break

        net.train()
        loss = net(prep(np.random.choice(d.N, size=args.n_batch)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss = loss.item()
        lr = scheduler._last_lr[0]
        log(f'loss={loss:.4f} lr={lr:.4f}', loss=loss, lr=lr)

        scheduler.step()
    writer.close()


# evaluate model
def evaluate(args, d, net, log=None):
    print(f'Evaluating on {d.N} problems...')
    eval_start_time = time()
    net.eval()
    prep = get_prepare_subproblem(d)
    total_loss = 0
    with torch.no_grad():
        for idxs in np.split(range(d.N), range(args.n_batch, d.N, args.n_batch)):
            loss = net(prep(idxs))
            total_loss += loss.item() * len(idxs)
    loss = total_loss / d.N
    eval_time = time() - eval_start_time

    if log is not None:
        log(f'eval_loss={loss:.4f} eval_time={eval_time:.1f}s', eval_loss=loss, eval_time=eval_time)


# Generate predictions
def generate(args, d, model):
    results = {"distance_preds": [], "distance_true": []}

    xs, offsets, node_idxs, n_subp_nodes = d.xs, d.offsets, d.node_idxs, d.n_subp_nodes
    data = Namespace(
        xs=xs, offsets=offsets, node_idxs=pad_each(node_idxs), n_subp_nodes=n_subp_nodes
    )

    with torch.no_grad():
        preds = model(get_prepare_subproblem()(data)).cpu().numpy()
    print(preds)

    labels = d['dists']
    if args.loss.startswith('MAE'):
        mae = mean_absolute_error(labels, preds)

    print(f'Mean absolute error for testing dataset is {mae}')

    n_correct = 0
    for i in range(len(labels) - 3):
        pred_idx = pd.Series(preds[i:i + 3]).idxmin()
        label_idx = pd.Series(labels[i:i + 3]).idxmin()
        if pred_idx == label_idx:
            n_correct += 1
    print(f'Correct percentage for predicting min cost every 3 allocations is {n_correct / (len(preds - 3))}')

    results["distance_preds"] = preds
    results["distance_true"] = labels

    args.generate_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.generate_save_dir / model_name
    pd.DataFrame(results).to_csv(save_path)


parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=Path)
parser.add_argument('train_dir', type=Path)
parser.add_argument('--data_suffix', type=str, default='')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--fit_subproblem', action='store_true')

# Parameters requiring tuning
parser.add_argument('--augment_rotate', action='store_true')
parser.add_argument('--augment_flip', action='store_true')
parser.add_argument('--augment_perturb_node', type=float, default=0.0)
parser.add_argument('--augment_perturb_route', type=float, default=0.0)
parser.add_argument('--n_steps', type=int, default=2000)
parser.add_argument('--n_step_save', type=int, default=500)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--n_batch', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=None)
parser.add_argument('--d_hidden', type=int, default=128)
parser.add_argument('--transformer_heads', type=int, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0)

parser.add_argument('--step', type=int, default=None)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--eval_partition', type=str, default='val', choices=['train', 'val', 'test'])
parser.add_argument('--n_step_eval', type=int, default=100)
parser.add_argument('--generate', action='store_true')
parser.add_argument('--save_suffix', type=str, default=None)
parser.add_argument('--generate_partition', type=str, default='val', choices=['train', 'val', 'test'])
parser.add_argument('--n_step_generate', type=int, default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--n_cpus', type=int, default=None)

type_map = {a.dest: a.type for a in parser._actions}

if __name__ == '__main__':
    args = parser.parse_args()
    args.train_dir.mkdir(parents=True, exist_ok=True)
    model_name = f'layer{args.n_layers}_head{args.transformer_heads}_lr{args.lr}_hidden{args.d_hidden}_batch{args.n_batch}_step{args.n_steps}'
    args.model_save_dir = args.train_dir / 'models' / model_name
    args.model_save_dir.mkdir(parents=True, exist_ok=True)
    args.generate_save_dir = args.train_dir / 'generations'

    config = args.model_save_dir / 'config.yaml'
    if args.eval or args.generate:
        assert config.exists()
        obj = load_yaml(config)
        for k, v in obj.items():
            if getattr(args, k) == parser.get_default(k):
                type_ = type_map[k]
                setattr(args, k, type_(v) if type_ is not None else v)
        print(f'Loaded args from {config}')
    else:
        obj = {k: v if isinstance(v, yaml_types) else str(v) for k, v in args.__dict__.items() if
               v != parser.get_default(k)}
        if config.exists():
            prev_obj = load_yaml(config)
            assert sorted(prev_obj.items()) == sorted(
                obj.items()), f'Previous training configuration at {config} is different than current training run\'s ' \
                              f'configs. Either use the same configs or delete {config.parent}.'
        else:
            save_yaml(config, obj)
            print(f'Saved args to {config}')
    print(args, flush=True)

    args.n_step_generate = args.n_step_generate or np.inf
    args.n_cpus = args.n_cpus

    if args.fit_subproblem:
        args.loss = args.loss or 'MAE'  # set loss function
        load, suffix = (load_subproblem_data, 'subproblems')
        path_eval = args.save_dir / f'{args.eval_partition}{args.data_suffix}_{suffix}.npz'
        d_eval = load(path_eval)
        print(f'Loaded evaluation data from {path_eval}. {d_eval.N} total labeled subproblems')

    d_generate = None
    if args.generate:
        load, suffix = (load_subproblem_data, 'subproblems')
        path_gen = args.save_dir / f'{args.generate_partition}{args.data_suffix}_{suffix}.npz'
        d_generate = load(path_gen)
        print(f'Loaded {d_generate.N} problems to generate predictions')

    if args.eval or args.generate:
        model = SubproblemNetwork(args, d_eval)
        step = restore(args, model)
        model = model.to(args.device)

        if args.eval:
            evaluate(args, d_eval, model, log=lambda *args, **kwargs: print(*args))

        if args.generate:
            generate(args, d_generate, model)
            # generate(args, d_generate, model)
    else:
        print(f'Saving experiment progress in {args.train_dir}')

        path_train = args.save_dir / f'train_{suffix}.npz'
        d = load(path_train)
        print(f'Loaded training data from {path_train}. {d.N} total labeled subproblems')
        # args.n_batch = d.N
        train(args, d, d_eval, d_generate)
