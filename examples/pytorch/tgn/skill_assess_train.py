import argparse
import traceback
import time
import copy

import numpy as np
import dgl
import torch

from tgn import TGN
from data_preprocess import TemporalWikipediaDataset, TemporalRedditDataset, TemporalDataset
from dataloading import (FastTemporalEdgeCollator, FastTemporalSampler,
                         SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                         TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.metrics import average_precision_score, roc_auc_score

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)


def train(model, dataloader, sampler, criterion, optimizer, args):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()

    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        optimizer.zero_grad()
        pred_pos, pred_neg = model.embed(
            positive_pair_g, negative_pair_g, blocks)
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * args.batch_size
        retain_graph = True if batch_cnt == 0 and not args.fast_mode else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.detach_memory()
        if not args.not_use_memory:
            model.update_memory(positive_pair_g)
        if args.fast_mode:
            sampler.attach_last_update(model.memory.last_update_t)
        print("Batch: ", batch_cnt, "Time: ", time.time() - last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss


def test_val(model, dataloader, sampler, criterion, args):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        for _, postive_pair_g, negative_pair_g, blocks in dataloader:
            pred_pos, pred_neg = model.embed(
                postive_pair_g, negative_pair_g, blocks)
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * batch_size
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            if not args.not_use_memory:
                model.update_memory(postive_pair_g)
            if args.fast_mode:
                sampler.attach_last_update(model.memory.last_update_t)
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50,
                        help='epochs for training on entire dataset')
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Size of each batch")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--memory_dim", type=int, default=100,
                        help="dimension of memory")
    parser.add_argument("--temporal_dim", type=int, default=100,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--memory_updater", type=str, default='gru',
                        help="Recurrent unit for memory update")
    parser.add_argument("--aggregator", type=str, default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--fast_mode", action="store_true", default=False,
                        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
    parser.add_argument("--simple_mode", action="store_true", default=False,
                        help="Simple Mode directly delete the temporal edges from the original static graph")
    parser.add_argument("--num_negative_samples", type=int, default=1,
                        help="number of negative samplers per positive samples")
    parser.add_argument("--dataset", type=str, default="wikipedia",
                        help="dataset selection wikipedia/reddit")
    parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
    parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")

    args = parser.parse_args()

    assert not (
            args.fast_mode and args.simple_mode), "you can only choose one sampling mode"

    # if args.k_hop != 1:
    #     assert args.simple_mode, "this k-hop parameter only support simple mode"

    gs, _ = dgl.load_graphs('/mnt/DS_SHARED/users/dvirb/data/research/graphs/games/pubg/small_ffa.bin')
    data = gs[0]

    # reverse the edges
    rel = data.edge_type_subgraph([('player', 'plays', 'match')])
    reverse_rel = dgl.reverse(rel, copy_edata=True)
    reversed_edges = reverse_rel.all_edges()
    data.add_edges(*reversed_edges, data=reverse_rel.edata, etype=('match', 'played_by', 'player'))

    # Pre-process data, mask new node in test set from original graph
    num_nodes = data.num_nodes()
    num_edges = data.num_edges(etype='plays')

    num_lasting_nodes = data.num_nodes('player')
    num_dispensable_nodes = data.num_nodes('match')

    # set split according to preset fraction
    # =======================================

    # train
    train_div = int(TRAIN_SPLIT * num_edges)
    train_last_ts = data.edges['plays'].data['timestamp'][train_div]

    # validation
    trainval_div = int(VALID_SPLIT * num_edges)
    # get split time
    div_time = data.edges['plays'].data['timestamp'][trainval_div]
    # update index according to time considerations
    trainval_div = (1 + (data.edges['plays'].data['timestamp'] <= div_time).nonzero()[
        -1]).item()  # get last index where time is prior (or equal) to div_time, and assign the next index (+1)

    # Select new node from test set and remove them from entire graph
    test_split_ts = data.edges['plays'].data['timestamp'][trainval_div]
    test_nodes = {
        'player': data.edges(etype='plays')[0][trainval_div:].unique().numpy(),
        'match': data.edges(etype='plays')[1][trainval_div:].unique().numpy()
    }

    test_new_nodes = {
        'player': np.random.choice(test_nodes['player'], int(0.1 * len(test_nodes['player'])), replace=False)}

    in_subg = dgl.in_subgraph(data, test_new_nodes)
    out_subg = dgl.out_subgraph(data, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = {
        etype: in_subg.edges[etype].data[dgl.EID][in_subg.edges[etype].data['timestamp'] < test_split_ts] for etype in
        data.canonical_etypes}  # for each edge type get the edge ids that precede `test_split_ts`
    new_node_out_eid_delete = {
        etype: out_subg.edges[etype].data[dgl.EID][out_subg.edges[etype].data['timestamp'] < test_split_ts] for etype in
        data.canonical_etypes}  # for each edge type get the edge ids that precede `test_split_ts`

    new_node_eid_delete = {etype: torch.cat(
        [new_node_in_eid_delete[etype], new_node_out_eid_delete[etype]]).unique() for etype in
                           data.canonical_etypes}

    graph_new_node = copy.deepcopy(data)
    # relative order preseved
    for etype, eid in new_node_eid_delete.items():
        graph_new_node.remove_edges(eid, etype)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = {
        etype: in_subg.edges[etype].data[dgl.EID] for etype in data.canonical_etypes
    }
    out_eid_delete = {
        etype: out_subg.edges[etype].data[dgl.EID] for etype in data.canonical_etypes
    }
    eid_delete = {
        etype: torch.cat([in_eid_delete[etype], out_eid_delete[etype]]).unique() for etype in data.canonical_etypes
    }

    graph_no_new_node = copy.deepcopy(data)
    for etype, eid in eid_delete.items():
        graph_no_new_node.remove_edges(eid, etype)

    # graph_no_new_node and graph_new_node should have same set of nid

    # Sampler Initialization
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(graph_no_new_node, fan_out)
        new_node_sampler = SimpleTemporalSampler(data, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(graph_no_new_node, k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(data, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors, hops=args.k_hop)
        edge_collator = TemporalEdgeCollator

    # Set Train, validation, test and new node test id
    train_bool = {etype: graph_no_new_node.edges[etype].data['timestamp'] <= train_last_ts for etype in
                  data.canonical_etypes}
    valid_bool = {etype: torch.logical_and(graph_no_new_node.edges[etype].data['timestamp'] > train_last_ts,
                                           graph_no_new_node.edges[etype].data['timestamp'] <= div_time) for etype in
                  data.canonical_etypes}
    test_bool = {etype: graph_no_new_node.edges[etype].data['timestamp'] > div_time for etype in data.canonical_etypes}
    test_new_node_bool = {etype: graph_new_node.edges[etype].data['timestamp'] > div_time for etype in
                          data.canonical_etypes}

    train_seed, valid_seed, test_seed, test_new_node_seed = dict(), dict(), dict(), dict()
    for etype in data.canonical_etypes:  # get eids
        train_seed[etype] = graph_no_new_node.edges('all', etype=etype)[-1][train_bool[etype]]
        valid_seed[etype] = graph_no_new_node.edges('all', etype=etype)[-1][valid_bool[etype]]
        test_seed[etype] = graph_no_new_node.edges('all', etype=etype)[-1][test_bool[etype]]
        test_new_node_seed[etype] = graph_new_node.edges('all', etype=etype)[-1][test_new_node_bool[etype]]

    # train_edges_num = (1 + (graph_no_new_node.edata['timestamp'] <= train_last_ts).nonzero()[-1]).item()
    # train_seed = torch.arange(train_edges_num)
    # valid_seed = torch.arange(train_edges_num, trainval_div - new_node_eid_delete.size(0))
    # test_seed = torch.arange(
    #     trainval_div - new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    # test_new_node_seed = torch.arange(
    #     trainval_div - new_node_eid_delete.size(0), graph_new_node.num_edges())

    # I STOPPED HERE
    g_sampling = None if args.fast_mode else graph_no_new_node
    new_node_g_sampling = None if args.fast_mode else graph_new_node
    if not args.fast_mode:
        for ntype in data.ntypes:
            new_node_g_sampling.nodes[ntype].data[dgl.NID] = new_node_g_sampling.nodes(ntype)
            g_sampling.nodes[ntype].data[dgl.NID] = new_node_g_sampling.nodes(ntype)

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    reverse_etypes = {
        ('player', 'plays', 'match'): ('match', 'played_by', 'player'),
        ('match', 'played_by', 'player'): ('player', 'plays', 'match')
    }
    train_dataloader = TemporalEdgeDataLoader(g=graph_no_new_node,
                                              eids=train_seed,
                                              graph_sampler=sampler,
                                              batch_size=args.batch_size * 2500,  # TODO: change back
                                              negative_sampler=None,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              # g_sampling=g_sampling,
                                              exclude='reverse_types',
                                              reverse_etypes=reverse_etypes)

    valid_dataloader = TemporalEdgeDataLoader(g=graph_no_new_node,
                                              eids=valid_seed,
                                              graph_sampler=sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=None,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              # g_sampling=g_sampling,
                                              exclude='reverse_types',
                                              reverse_etypes=reverse_etypes)

    test_dataloader = TemporalEdgeDataLoader(g=graph_no_new_node,
                                             eids=test_seed,
                                             graph_sampler=sampler,
                                             batch_size=args.batch_size,
                                             negative_sampler=None,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             # g_sampling=g_sampling,
                                             exclude='reverse_types',
                                             reverse_etypes=reverse_etypes)

    test_new_node_dataloader = TemporalEdgeDataLoader(g=graph_new_node,
                                                      eids=test_new_node_seed,
                                                      graph_sampler=new_node_sampler if args.fast_mode else sampler,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=None,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      # g_sampling=new_node_g_sampling,
                                                      exclude='reverse_types',
                                                      reverse_etypes=reverse_etypes)

    etypes = data.canonical_etypes
    assert data.edata['feats'][etypes[0]].shape[1] == data.edata['feats'][etypes[1]].shape[1]
    edge_dim = data.edata['feats'][etypes[0]].shape[1]

    model = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_nodes=num_lasting_nodes,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop)

    criterion = torch.nn.L1Loss()  # Should be changed to informational loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Implement Logging mechanism
    f = open("logging.txt", 'w')
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            train_loss = train(model, train_dataloader, sampler,
                               criterion, optimizer, args)

            val_ap, val_auc = test_val(
                model, valid_dataloader, sampler, criterion, args)
            memory_checkpoint = model.store_memory()
            if args.fast_mode:
                new_node_sampler.sync(sampler)
            test_ap, test_auc = test_val(
                model, test_dataloader, sampler, criterion, args)
            model.restore_memory(memory_checkpoint)
            if args.fast_mode:
                sample_nn = new_node_sampler
            else:
                sample_nn = sampler
            nn_test_ap, nn_test_auc = test_val(
                model, test_new_node_dataloader, sample_nn, criterion, args)
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                i, train_loss, val_ap, val_auc))
            log_content.append(
                "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
            log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                i, nn_test_ap, nn_test_auc))

            f.writelines(log_content)
            model.reset_memory()
            if i < args.epochs - 1 and args.fast_mode:
                sampler.reset()
            print(log_content[0], log_content[1], log_content[2])
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
