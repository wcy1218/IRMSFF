import os
import random
import pandas as pd
import numpy
from nlgeval import compute_metrics
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

from second_stage_train.utils import read_examples, convert_examples_to_features

import torch.nn.functional as F
import torch.nn as nn
import torch

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Batch

class ASTGNN(nn.Module):
    """
    图神经网络用于提取AST特征
    """
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_type='gcn', num_layers=2, dropout=0.1):
        super(ASTGNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 节点类型和token的嵌入层
        self.node_type_embedding = nn.Embedding(100, 64)  # 假设最多100种节点类型
        self.token_embedding = nn.Embedding(50000, 64)    # 词汇表大小
        
        # 输入投影层
        self.input_proj = nn.Linear(128, input_dim)
        
        # 使用更兼容的GNN层
        if gnn_type == 'gcn':
            # 使用简单的GraphConv替代GCNConv
            from torch_geometric.nn import GraphConv
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                in_channels = input_dim if i == 0 else hidden_dim
                out_channels = hidden_dim if i < num_layers - 1 else output_dim
                conv = GraphConv(in_channels, out_channels)
                self.convs.append(conv)
                
        elif gnn_type == 'gat':
            # 使用GATv2Conv替代GATConv（更稳定）
            try:
                from torch_geometric.nn import GATv2Conv
                self.convs = nn.ModuleList()
                for i in range(num_layers):
                    in_channels = input_dim if i == 0 else hidden_dim
                    out_channels = hidden_dim if i < num_layers - 1 else output_dim
                    heads = 4
                    concat = (i < num_layers - 1)
                    conv = GATv2Conv(in_channels, out_channels // (heads if concat else 1), 
                                    heads=heads, concat=concat)
                    self.convs.append(conv)
            except ImportError:
                # 回退到GraphConv
                print("GATv2Conv不可用，使用GraphConv替代")
                from torch_geometric.nn import GraphConv
                self.convs = nn.ModuleList()
                for i in range(num_layers):
                    in_channels = input_dim if i == 0 else hidden_dim
                    out_channels = hidden_dim if i < num_layers - 1 else output_dim
                    conv = GraphConv(in_channels, out_channels)
                    self.convs.append(conv)
        
        # 批归一化层
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
        
        # 输出层
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, ast_data):
        """
        ast_data: 源代码的AST图数据
        应包含: 
        - node_types: 节点类型 [num_nodes]
        - node_tokens: 节点token [num_nodes]
        - edge_index: 边索引 [2, num_edges]
        - batch: 批索引 [num_nodes]
        """
        # 获取节点特征
        node_type_emb = self.node_type_embedding(ast_data['node_types'])
        token_emb = self.token_embedding(ast_data['node_tokens'])
        x = torch.cat([node_type_emb, token_emb], dim=-1)
        x = self.input_proj(x)
        
        # GNN消息传递
        edge_index = ast_data['edge_index']
        batch = ast_data['batch']
        
        for i in range(self.num_layers):
            if i > 0:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if i < self.num_layers - 1:
                    x = self.bns[i-1](x)
            
            conv = self.convs[i]
            x = conv(x, edge_index)
        
        # 全局池化得到图级别表示
        graph_emb = global_mean_pool(x, batch)
        graph_emb = self.output_proj(graph_emb)
        
        return graph_emb
    
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None, l2_norm=False, fusion=True, use_ast=False, ast_hidden_dim=256, ast_output_dim=256, gnn_type='gcn'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.orgencoder = RobertaModel.from_pretrained("../../model/unixcoder")
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        self.l2_norm = l2_norm
        self.fusion_flag = fusion
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.use_ast = use_ast
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        
        # AST GNN
        if self.use_ast:
            self.ast_gnn = ASTGNN(
                input_dim=config.hidden_size,  # 与编码器输出维度一致
                hidden_dim=ast_hidden_dim,
                output_dim=ast_output_dim,
                gnn_type=gnn_type
            )
        
        if(self.fusion_flag==True):
            fusion_input_dim = config.hidden_size * 2
            self.fusion1 = nn.Linear(fusion_input_dim, config.hidden_size)
            self.fusion2 = nn.Linear(fusion_input_dim, config.hidden_size)
            self.fusion3 = nn.Linear(fusion_input_dim, config.hidden_size)
            self.fusion = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            self.fusion_single = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, similarity_ids=None, similarity_mask=None, target_ids=None, target_mask=None, source_ast_data=None):
        source_outputs = self.encoder(source_ids, attention_mask=source_mask)
        source_encoder_output = source_outputs[0].permute([1, 0, 2]).contiguous()
        similarity_outputs = self.orgencoder(similarity_ids, attention_mask=similarity_mask)
        similarity_encoder_output = similarity_outputs[0].permute([1, 0, 2]).contiguous()
        
            # 提取源代码AST特征
        ast_features = None
        if self.use_ast and source_ast_data is not None:
            source_ast_emb = self.ast_gnn(source_ast_data)
            # source_ast_emb: [batch_size, ast_output_dim]

            # 确保AST维度与hidden_dim一致
            if source_ast_emb.size(-1) != self.config.hidden_size:
                if not hasattr(self, 'ast_projection'):
                    self.ast_projection = nn.Linear(source_ast_emb.size(-1), 
                                                    self.config.hidden_size).to(source_ast_emb.device)
                source_ast_emb = self.ast_projection(source_ast_emb)

            # 扩展维度以匹配序列长度 [seq_len, batch_size, hidden_dim]
            # seq_len = source_encoder_output.size(0)
            # ast_features = source_ast_emb.unsqueeze(0).repeat(seq_len, 1, 1)
            seq_len = source_encoder_output.size(0)
            batch_size = source_ast_emb.size(0)
            hidden_dim = source_ast_emb.size(1)

            # 从[batch_size, hidden_dim]变为[seq_len, batch_size, hidden_dim]
            ast_features = source_ast_emb.unsqueeze(0)  # [1, batch_size, hidden_dim]
            ast_features = ast_features.expand(seq_len, -1, -1)  # [seq_len, batch_size, hidden_dim]
            
        # encoder_output [seq_len, batch_size, d_model]
        # fusion
        if(self.l2_norm==True):
            source_encoder_output = F.normalize(source_encoder_output, p=2, dim=-1)
            similarity_encoder_output = F.normalize(similarity_encoder_output, p=2, dim=-1)
            if ast_features is not None:
                ast_features = F.normalize(ast_features, p=2, dim=-1)
        if(self.fusion_flag==True):
            if self.use_ast and ast_features is not None:
                # ========== 第一层：源代码与AST门控融合 ==========
                if not hasattr(self, 'gated_fusion'):
                    # 门控机制
                    self.gate_layer = nn.Sequential(
                        nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                        nn.Sigmoid()  # 输出0-1的权重
                    ).to(source_encoder_output.device)

                    # 特征转换层
                    self.feature_transform = nn.Sequential(
                        nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                        nn.ReLU(),
                        nn.Dropout(self.config.hidden_dropout_prob)
                    ).to(source_encoder_output.device)

                if self.use_ast and ast_features is not None:
                    # 扩展AST特征维度（如果需要）
                    if ast_features.size(0) != source_encoder_output.size(0):
                        ast_features = ast_features.expand(source_encoder_output.size(0), -1, -1)

                    # 拼接特征
                    combined = torch.cat([source_encoder_output, ast_features], dim=-1)

                    # 计算门控权重：决定保留多少原始代码特征和AST特征
                    gate_weights = self.gate_layer(combined)

                    # 门控融合：gate控制AST的贡献，(1-gate)控制代码的贡献
                    gated_fusion = gate_weights * ast_features + (1 - gate_weights) * source_encoder_output

                    # 特征变换
                    enhanced_code = self.feature_transform(combined) + gated_fusion  # 残差连接
                    enhanced_code = torch.relu(enhanced_code)
                else:
                    enhanced_code = source_encoder_output
                # ========== 第二层：基于注意力的代码-注释融合 ==========
                # 1. 创建注意力层（如果不存在）
                if not hasattr(self, 'attention_fusion_layer'):
                    # 注意力机制：代码作为query，注释作为key/value
                    self.attention_fusion_layer = nn.MultiheadAttention(
                        embed_dim=self.config.hidden_size,
                        num_heads=8,  # 可以使用 config.num_attention_heads
                        dropout=self.config.hidden_dropout_prob,
                        batch_first=False  # 注意：你的数据是 [seq_len, batch, hidden]
                    ).to(source_encoder_output.device)

                    # 注意力后的融合层
                    self.post_attention_fusion = nn.Sequential(
                        nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                        nn.ReLU(),
                        nn.Dropout(self.config.hidden_dropout_prob),
                        nn.Linear(self.config.hidden_size, self.config.hidden_size)
                    ).to(source_encoder_output.device)

                # 2. 计算注意力：增强的代码关注注释
                # enhanced_code: [seq_len1, batch, hidden]
                # similarity_encoder_output: [seq_len2, batch, hidden]

                attended_output, attention_weights = self.attention_fusion_layer(
                    query=enhanced_code,  # query: 增强的代码表示
                    key=similarity_encoder_output,  # key: 注释特征
                    value=similarity_encoder_output,  # value: 注释特征
                    key_padding_mask=None  # 可以传入相似性掩码
                )
#                 # 方案1：保持原始三路融合，但用AST增强
#                 output_1 = self.fusion1(torch.cat([
#                     source_encoder_output, 
#                     similarity_encoder_output
#                 ], dim=-1))

#                 # 用AST替代或增强原始的第二路
#                 output_2 = self.fusion2(torch.cat([
#                     source_encoder_output, 
#                     ast_features
#                 ], dim=-1))

#                 # 用AST增强原始第三路
#                 output_3 = self.fusion3(torch.cat([
#                     source_encoder_output * similarity_encoder_output,
#                     ast_features
#                 ], dim=-1))
                if not hasattr(self, 'attention_gate'):
                    self.attention_gate = nn.Sequential(
                        nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                        nn.Sigmoid()
                    ).to(source_encoder_output.device)

                gate_input = torch.cat([enhanced_code, attended_output], dim=-1)
                fusion_gate = self.attention_gate(gate_input)
                encoder_output = fusion_gate * enhanced_code + (1 - fusion_gate) * attended_output
            else:
                # 原始双特征融合
                output_1 = self.fusion1(torch.cat([source_encoder_output, similarity_encoder_output], dim=-1))
                output_2 = self.fusion2(torch.cat([source_encoder_output, source_encoder_output - similarity_encoder_output], dim=-1))
                output_3 = self.fusion3(torch.cat([source_encoder_output, source_encoder_output * similarity_encoder_output], dim=-1))
            
                out_put = torch.cat([output_1, output_2, output_3], dim=-1)
                encoder_output = self.dropout(self.fusion(out_put))
        else:
            encoder_output = self.dropout(self.fusion_single(torch.cat([source_encoder_output, similarity_encoder_output], dim=-1)))
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
    
class ASTProcessor:
    """
    简化的AST数据处理器 - 支持按需解析
    """
    def __init__(self, tokenizer, max_nodes=200):
        self.tokenizer = tokenizer
        self.max_nodes = max_nodes
        self.type_vocab = self._create_type_vocab()
        
        # AST缓存
        # self.ast_cache = {}
        
    def _create_type_vocab(self):
        """创建节点类型词汇表"""
        type_vocab = {}
        # 基础节点类型
        base_types = [
            'CompilationUnit', 'ClassDeclaration', 'MethodDeclaration',
            'VariableDeclaration', 'Assignment', 'BinaryOperation',
            'Literal', 'Identifier', 'IfStatement', 'ForStatement',
            'WhileStatement', 'ReturnStatement', 'TryStatement',
            'CatchClause', 'ThrowStatement', 'MethodInvocation',
            'FieldAccess', 'This', 'Super', 'Annotation',
            'FormalParameter', 'MemberReference', 'TypeDeclaration',
            'BlockStatement', 'LocalVariableDeclaration', 'ExpressionStatement'
        ]
        
        for i, type_name in enumerate(base_types):
            type_vocab[type_name] = i
        
        # 未知类型
        type_vocab['UNKNOWN'] = len(base_types)
        
        return type_vocab
    
    def parse_single_java(self, java_code, idx=None):
        """解析单个Java代码的AST - 按需解析"""
        def wrap(code):
            return f"""
                public class M {{
                    {code}
                }}
                """
        # if idx is not None and idx in self.ast_cache:
        #     return self.ast_cache[idx]
        
        try:
            # 尝试使用javalang
            from javalang import parse
            tree = parse.parse(wrap(java_code))
            ast_data = self._extract_from_javalang(tree)
        except Exception as e:
            # 如果解析失败，使用简单规则
            # print(f"AST解析失败，使用简单规则: {e}")
            ast_data = self._parse_with_simple_rules(java_code)
        
        # 缓存结果
        # if idx is not None:
        #     self.ast_cache[idx] = ast_data
            
        return ast_data
        

    def _extract_from_javalang(self, tree):
        """从javalang AST提取数据"""
        nodes = []
        edges = []
        node_types = []
        node_tokens = []
        
        node_counter = 0
        parent_stack = []
        
        def visit_node(node, depth=0, parent_idx=-1):
            nonlocal node_counter
            if node_counter >= self.max_nodes or depth > 10:  # 限制深度
                return
            
            current_idx = node_counter
            nodes.append(current_idx)
            node_counter += 1
            
            # 获取节点类型
            node_type = type(node).__name__
            node_types.append(self._get_type_id(node_type))
            
            # 获取节点文本
            token = self._get_node_token(node)
            token_id = self.tokenizer.encode(token, add_special_tokens=False, max_length=1, truncation=True)
            node_tokens.append(token_id[0] if token_id else 0)
            
            # 添加边
            if parent_idx != -1:
                edges.append([parent_idx, current_idx])
            
            # 递归处理子节点
            for child in self._get_children(node):
                if child is not None:
                    visit_node(child, depth + 1, current_idx)
        
        try:
            visit_node(tree)
        except Exception as e:
            print(f"AST遍历错误: {e}")
            return self._create_default_ast()
        
        # 转换为张量
        if len(nodes) == 0:
            return self._create_default_ast()
            
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # 如果没有边，创建一个简单连接
            if len(nodes) > 1:
                edge_index = torch.tensor([[0], [1]], dtype=torch.long)
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)
        
        node_types = torch.tensor(node_types, dtype=torch.long)
        node_tokens = torch.tensor(node_tokens, dtype=torch.long)
        
        return {
            'node_types': node_types,
            'node_tokens': node_tokens,
            'edge_index': edge_index,
            'num_nodes': len(nodes)
        }
    
    def _parse_with_simple_rules(self, java_code):
        """使用简单规则解析Java代码"""
        import re
        
        nodes = []
        node_types = []
        node_tokens = []
        
        # 查找关键结构
        patterns = [
            (r'class\s+(\w+)', 'ClassDeclaration'),
            (r'(\w+)\s*\([^)]*\)\s*\{', 'MethodDeclaration'),
            (r'(\w+)\s*=\s*[^;]+;', 'VariableDeclaration'),
            (r'if\s*\(', 'IfStatement'),
            (r'for\s*\(', 'ForStatement'),
            (r'while\s*\(', 'WhileStatement'),
            (r'return\s+', 'ReturnStatement'),
            (r'new\s+(\w+)', 'MethodInvocation'),
        ]
        
        node_counter = 0
        for pattern, node_type in patterns:
            matches = re.findall(pattern, java_code[:1000])  # 只检查前1000个字符
            for match in matches:
                if node_counter >= self.max_nodes:
                    break
                    
                nodes.append(node_counter)
                node_types.append(self._get_type_id(node_type))
                
                if isinstance(match, tuple):
                    token_text = match[0] if match else node_type
                else:
                    token_text = match if match else node_type
                    
                token_id = self.tokenizer.encode(token_text, add_special_tokens=False, max_length=1, truncation=True)
                node_tokens.append(token_id[0] if token_id else 0)
                node_counter += 1
        
        if len(nodes) == 0:
            return self._create_default_ast()
        
        # 创建简单边（树结构）
        edges = []
        for i in range(1, min(len(nodes), 10)):  # 限制边数量
            parent = i - 1
            edges.append([parent, i])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        node_types = torch.tensor(node_types, dtype=torch.long)
        node_tokens = torch.tensor(node_tokens, dtype=torch.long)
        
        return {
            'node_types': node_types,
            'node_tokens': node_tokens,
            'edge_index': edge_index,
            'num_nodes': len(nodes)
        }
    
    def _get_type_id(self, node_type):
        """获取节点类型ID"""
        return self.type_vocab.get(node_type, self.type_vocab['UNKNOWN'])
    
    def _get_node_token(self, node):
        """获取节点文本"""
        if hasattr(node, 'name'):
            return str(node.name)
        elif hasattr(node, 'value'):
            return str(node.value)
        elif hasattr(node, 'token') and node.token is not None:
            return str(node.token.value) if hasattr(node.token, 'value') else str(node.token)
        else:
            return type(node).__name__
    
    def _get_children(self, node):
        """获取子节点"""
        children = []
        if hasattr(node, 'children'):
            children.extend(node.children)
        if hasattr(node, 'body'):
            if isinstance(node.body, list):
                children.extend(node.body)
            elif node.body:
                children.append(node.body)
        return children
    
    def _create_default_ast(self):
        """创建默认AST数据"""
        num_nodes = 3
        node_types = torch.tensor([self.type_vocab['ClassDeclaration'], 
                                 self.type_vocab['MethodDeclaration'], 
                                 self.type_vocab['Identifier']], dtype=torch.long)
        node_tokens = torch.tensor([101, 102, 103], dtype=torch.long)  # 简单的token
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        
        return {
            'node_types': node_types,
            'node_tokens': node_tokens,
            'edge_index': edge_index,
            'num_nodes': num_nodes
        }
    def prepare_batch_ast_data(self, batch_codes):
        """批量准备AST数据 - 适配现有代码"""
        print(f"正在批量处理 {len(batch_codes)} 个源代码的AST...")
        batch_ast_dicts = []
        
        for i, code in enumerate(batch_codes):
            if i % 100 == 0 and i > 0:
                print(f"  已处理 {i}/{len(batch_codes)} 个代码")
            
            # 使用现有的parse_single_java方法
            ast_data = self.parse_single_java(code, idx=i)
            batch_ast_dicts.append(ast_data)
        
        print(f"AST数据准备完成: {len(batch_ast_dicts)} 个AST图")
        return self._batch_ast_dicts(batch_ast_dicts)
    
    def _batch_ast_dicts(self, ast_dicts):
        """将多个AST字典批处理为一个"""
        if not ast_dicts:
            return None
        
        all_node_types = []
        all_node_tokens = []
        all_edge_indices = []
        all_batches = []
        
        node_offset = 0
        for i, ast_dict in enumerate(ast_dicts):
            if ast_dict is None:
                continue
                
            num_nodes = ast_dict['num_nodes']
            
            # 收集节点特征
            all_node_types.append(ast_dict['node_types'])
            all_node_tokens.append(ast_dict['node_tokens'])
            
            # 调整边索引
            if ast_dict['edge_index'].numel() > 0:
                adjusted_edges = ast_dict['edge_index'] + node_offset
                all_edge_indices.append(adjusted_edges)
            
            # 批索引
            batch_idx = torch.full((num_nodes,), i, dtype=torch.long)
            all_batches.append(batch_idx)
            
            node_offset += num_nodes
        
        if not all_node_types:
            return None
        
        # 合并所有数据
        node_types = torch.cat(all_node_types, dim=0)
        node_tokens = torch.cat(all_node_tokens, dim=0)
        batch = torch.cat(all_batches, dim=0)
        
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        return {
            'node_types': node_types,
            'node_tokens': node_tokens,
            'edge_index': edge_index,
            'batch': batch,
            'num_graphs': len(ast_dicts)
        }
    
    def parse_java_to_ast(self, java_code):
        """兼容旧代码的方法名 - 调用parse_single_java"""
        return self.parse_single_java(java_code)
    
class My_model():
    def __init__(self, codebert_path, decoder_layers, fix_encoder, beam_size, max_source_length, max_target_length, load_model_path, l2_norm, fusion, use_ast=False, gnn_type='gcn'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained(codebert_path)
        self.tokenizer = tokenizer_class.from_pretrained(codebert_path)
        # length config
        self.max_source_length, self.max_target_length = max_source_length, max_target_length
        self.beam_size = beam_size
        self.fusion = fusion
        self.use_ast = use_ast
        self.gnn_type = gnn_type
        
        # AST处理器（只处理源代码）
        if self.use_ast:
            self.ast_processor = ASTProcessor(self.tokenizer)
        
        # build model
        encoder = model_class.from_pretrained(codebert_path)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=beam_size, max_length=max_target_length,
                        sos_id=self.tokenizer.cls_token_id, eos_id=self.tokenizer.sep_token_id, l2_norm=l2_norm, fusion=fusion, use_ast=use_ast,
            gnn_type=gnn_type)
        if load_model_path is not None:
            print("从...{}...重新加载参数".format(load_model_path))
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(load_model_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        print(self.model)
        if fix_encoder:
            encoder.requires_grad_(False)
            print("冻结Codebert的参数")
        self.model.to(self.device)
        if self.use_ast:
            print(f"启用AST GNN特征，使用 {gnn_type.upper()} 架构")
        
    def prepare_ast_data(self, batch_codes):
        """批量准备AST数据 - 修改为使用ASTProcessor的prepare_batch_ast_data"""
        if not hasattr(self, 'ast_processor'):
            return None
        
        return self.ast_processor.prepare_batch_ast_data(batch_codes)
    
    def extract_batch_ast_data(self, ast_batch, batch_indices):
        """从批处理AST数据中提取指定批次 - 修复版本"""
        if ast_batch is None or len(batch_indices) == 0:
            return None

        # 筛选属于当前批次的节点
        batch_tensor = ast_batch['batch']
        if batch_tensor.numel() == 0:
            return None

        # 创建掩码
        node_mask = torch.zeros(batch_tensor.max().item() + 1, dtype=torch.bool)
        node_mask[batch_indices] = True
        selected_nodes = node_mask[batch_tensor]

        if not selected_nodes.any():
            return None

        result = {
            'node_types': ast_batch['node_types'][selected_nodes],
            'node_tokens': ast_batch['node_tokens'][selected_nodes],
        }

        # 处理边索引
        edge_index = ast_batch['edge_index']
        if edge_index.numel() > 0:
            edge_mask = selected_nodes[edge_index[0]] & selected_nodes[edge_index[1]]
            filtered_edges = edge_index[:, edge_mask]

            if filtered_edges.numel() > 0:
                # 重新映射节点索引
                old_to_new = torch.full((len(selected_nodes),), -1, dtype=torch.long)
                old_to_new[selected_nodes] = torch.arange(selected_nodes.sum())

                result['edge_index'] = old_to_new[filtered_edges]
            else:
                result['edge_index'] = torch.tensor([[], []], dtype=torch.long)
        else:
            result['edge_index'] = torch.tensor([[], []], dtype=torch.long)

        original_batch = batch_tensor[selected_nodes]
        unique_batches = torch.unique(original_batch)

        # 重新映射批次索引：例如 [2, 2, 5, 5] -> [0, 0, 1, 1]
        batch_mapping = {int(old): new for new, old in enumerate(unique_batches)}
        new_batch = torch.tensor([batch_mapping[int(b)] for b in original_batch], dtype=torch.long)

        result['batch'] = new_batch

        return result
    
    def _batch_ast_data_simple(self, ast_data_list):
        """简化的AST数据批处理"""
        if not ast_data_list:
            return None
        
        # 限制每个图的节点数
        max_nodes_per_graph = 50
        all_node_types = []
        all_node_tokens = []
        all_edge_indices = []
        all_batches = []
        
        node_offset = 0
        for i, ast_data in enumerate(ast_data_list):
            if ast_data is None:
                continue
            
            # 限制节点数量
            num_nodes = min(ast_data['num_nodes'], max_nodes_per_graph)
            
            # 收集节点特征
            all_node_types.append(ast_data['node_types'][:num_nodes])
            all_node_tokens.append(ast_data['node_tokens'][:num_nodes])
            
            # 处理边索引
            if ast_data['edge_index'].numel() > 0:
                # 只保留有效的边
                valid_edges = ast_data['edge_index'][:, ast_data['edge_index'][0] < num_nodes]
                valid_edges = valid_edges[:, valid_edges[1] < num_nodes]
                
                if valid_edges.numel() > 0:
                    adjusted_edges = valid_edges + node_offset
                    all_edge_indices.append(adjusted_edges)
            
            # 批索引
            batch_idx = torch.full((num_nodes,), i, dtype=torch.long)
            all_batches.append(batch_idx)
            
            node_offset += num_nodes
        
        if not all_node_types:
            return None
        
        # 合并所有数据
        node_types = torch.cat(all_node_types, dim=0)
        node_tokens = torch.cat(all_node_tokens, dim=0)
        batch = torch.cat(all_batches, dim=0)
        
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        return {
            'node_types': node_types,
            'node_tokens': node_tokens,
            'edge_index': edge_index,
            'batch': batch
        }
    def train(self, train_filename, train_batch_size, num_train_epochs, learning_rate,
              do_eval, dev_filename, eval_batch_size, output_dir, gradient_accumulation_steps=1):
        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length, self.max_target_length, stage='train')
            
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_similarity_ids = torch.tensor([f.similarity_ids for f in train_features], dtype=torch.long)
        all_similarity_mask = torch.tensor([f.similarity_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        
        # 创建数据集
        dataset_size = len(train_examples)
            # 创建自定义数据集，按需解析AST
        class StreamASTDataset(torch.utils.data.Dataset):
            def __init__(self, source_ids, source_mask, similarity_ids, similarity_mask,
                        target_ids, target_mask, examples, ast_processor):
                self.source_ids = source_ids
                self.source_mask = source_mask
                self.similarity_ids = similarity_ids
                self.similarity_mask = similarity_mask
                self.target_ids = target_ids
                self.target_mask = target_mask
                self.examples = examples
                self.ast_processor = ast_processor

            def __len__(self):
                return len(self.source_ids)

            def __getitem__(self, idx):
                # 按需解析AST
                source_code = self.examples[idx].source
                ast_data = self.ast_processor.parse_single_java(source_code, idx)

                return (
                    self.source_ids[idx],
                    self.source_mask[idx],
                    self.similarity_ids[idx],
                    self.similarity_mask[idx],
                    self.target_ids[idx],
                    self.target_mask[idx],
                    ast_data
                )

        # 创建数据加载器
        if self.use_ast:
            train_dataset = StreamASTDataset(
                all_source_ids, all_source_mask,
                all_similarity_ids, all_similarity_mask,
                all_target_ids, all_target_mask,
                train_examples,
                self.ast_processor
            )

            # 自定义collate函数
            def ast_collate_fn(batch):
                source_ids = torch.stack([item[0] for item in batch])
                source_mask = torch.stack([item[1] for item in batch])
                similarity_ids = torch.stack([item[2] for item in batch])
                similarity_mask = torch.stack([item[3] for item in batch])
                target_ids = torch.stack([item[4] for item in batch])
                target_mask = torch.stack([item[5] for item in batch])
                ast_data_list = [item[6] for item in batch]

                # 批量处理AST数据
                batched_ast_data = self._batch_ast_data_simple(ast_data_list)

                return source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask, batched_ast_data
    

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_batch_size // gradient_accumulation_steps,
                shuffle=True,
                collate_fn=ast_collate_fn
            )
        else:
            # 不使用AST，使用原始TensorDataset
            train_data = TensorDataset(
                all_source_ids, all_source_mask,
                all_similarity_ids, all_similarity_mask,
                all_target_ids, all_target_mask
            )

            train_dataloader = DataLoader(
                train_data,
                batch_size=train_batch_size // gradient_accumulation_steps,
                shuffle=True
            )

        num_train_optimization_steps = -1

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)

        # Start training
        print("***** 开始训练 *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", train_batch_size)
        print("  Num epoch = %d", num_train_epochs)
        self.model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        
        def _streaming_evaluate(dev_filename, eval_batch_size, output_dir,
                          epoch, global_step, best_bleu, best_loss, dev_dataset):
            """流式评估 - 内部函数"""
            # 读取评估数据
            eval_examples = read_examples(dev_filename)

            # 创建评估数据集
            eval_features = convert_examples_to_features(eval_examples, self.tokenizer, 
                                                        self.max_source_length, 
                                                        self.max_target_length, stage='dev')

            # 创建张量
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_similarity_ids = torch.tensor([f.similarity_ids for f in eval_features], dtype=torch.long)
            all_similarity_mask = torch.tensor([f.similarity_mask for f in eval_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)

            # 流式评估数据集
            if self.use_ast:
                eval_dataset = StreamASTDataset(
                    all_source_ids, all_source_mask,
                    all_similarity_ids, all_similarity_mask,
                    all_target_ids, all_target_mask,
                    eval_examples,
                    self.ast_processor
                )

                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=ast_collate_fn,
                    num_workers=0
                )
            else:
                eval_data = TensorDataset(
                    all_source_ids, all_source_mask,
                    all_similarity_ids, all_similarity_mask,
                    all_target_ids, all_target_mask
                )

                eval_dataloader = DataLoader(
                    eval_data,
                    batch_size=eval_batch_size,
                    shuffle=False
                )

            print("\n***** Running evaluation *****")
            print("  epoch = %d", epoch)
            print("  Num examples = %d", len(eval_examples))
            print("  Batch size = %d", eval_batch_size)

            # Start Evaling model
            self.model.eval()
            eval_loss, tokens_num = 0, 0
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                if self.use_ast:
                    source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask, source_ast_data = batch
                    source_ids = source_ids.to(self.device)
                    source_mask = source_mask.to(self.device)
                    similarity_ids = similarity_ids.to(self.device)
                    similarity_mask = similarity_mask.to(self.device)
                    target_ids = target_ids.to(self.device)
                    target_mask = target_mask.to(self.device)
                    if source_ast_data:
                        source_ast_data = {k: v.to(self.device) for k, v in source_ast_data.items()}

                    with torch.no_grad():
                        _, loss, num = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids,
                            similarity_mask=similarity_mask, 
                            target_ids=target_ids, 
                            target_mask=target_mask,
                            source_ast_data=source_ast_data
                        )
                else:
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids,
                            similarity_mask=similarity_mask, 
                            target_ids=target_ids, 
                            target_mask=target_mask
                        )

                eval_loss += loss.sum().item()
                tokens_num += num.sum().item()

            # Print loss of dev dataset
            self.model.train()
            eval_loss = eval_loss / tokens_num
            result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                      'global_step': global_step + 1,
                      'train_loss': round(train_loss, 5)}
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
            print("  " + "*" * 20)

            # save last checkpoint
            last_output_dir = os.path.join(output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

            if eval_loss < best_loss:
                print("  Best ppl:%s", round(numpy.exp(eval_loss), 5))
                print("  " + "*" * 20)
                best_loss = eval_loss
                # Save best checkpoint for best ppl
                output_dir_ppl = os.path.join(output_dir, 'checkpoint-best-ppl')
                if not os.path.exists(output_dir_ppl):
                    os.makedirs(output_dir_ppl)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            # Calculate bleu
            eval_examples = read_examples(dev_filename)
            eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
            eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                         self.max_target_length, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_similarity_ids = torch.tensor([f.similarity_ids for f in eval_features], dtype=torch.long)
            all_similarity_mask = torch.tensor([f.similarity_mask for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_source_ids, all_source_mask, all_similarity_ids, all_similarity_mask)
            dev_dataset['dev_bleu'] = eval_examples, eval_data

            # 如果使用AST，为BLEU计算准备AST数据
            bleu_ast_data = None
            if self.use_ast:
                print("准备BLEU计算AST数据...")
                bleu_source_codes = [ex.source for ex in eval_examples]
                bleu_ast_data = self.ast_processor.prepare_batch_ast_data(bleu_source_codes)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

            self.model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, similarity_ids, similarity_mask = batch

                # 为当前批次准备AST数据
                batch_ast_data = None
                if self.use_ast and bleu_ast_data is not None:
                    # 从批处理AST数据中提取当前批次的AST数据
                    batch_indices = [i for i in range(len(batch[0]))]
                    batch_ast_data = self.extract_batch_ast_data(bleu_ast_data, batch_indices)
                    if batch_ast_data:
                        batch_ast_data = {k: v.to(self.device) for k, v in batch_ast_data.items()}

                with torch.no_grad():
                    if self.use_ast and batch_ast_data is not None:
                        # 修复AST特征维度
                        if batch_ast_data is not None:
                            # 确保AST数据有正确的结构
                            if isinstance(batch_ast_data, dict):
                                # 检查节点数量
                                if 'node_types' in batch_ast_data and batch_ast_data['node_types'].numel() > 0:
                                    # AST数据有效，传递给模型
                                    preds = self.model(
                                        source_ids=source_ids, 
                                        source_mask=source_mask, 
                                        similarity_ids=similarity_ids, 
                                        similarity_mask=similarity_mask,
                                        source_ast_data=batch_ast_data
                                    )
                                else:
                                    print("无效")
                                    # AST数据无效，跳过
                                    preds = self.model(
                                        source_ids=source_ids, 
                                        source_mask=source_mask, 
                                        similarity_ids=similarity_ids, 
                                        similarity_mask=similarity_mask
                                    )
                            else:
                                # batch_ast_data不是字典，跳过
                                preds = self.model(
                                    source_ids=source_ids, 
                                    source_mask=source_mask, 
                                    similarity_ids=similarity_ids, 
                                    similarity_mask=similarity_mask
                                )
                    else:
                        preds = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids, 
                            similarity_mask=similarity_mask
                        )

                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)

            self.model.train()

            csv_pred_list = []
            csv_true_list = []

            for ref, gold in zip(p, eval_examples):
                csv_pred_list.append(gold.target)
                csv_true_list.append(ref)

            df = pd.DataFrame(csv_true_list)
            df.to_csv(os.path.join(output_dir, "valid_hyp.csv"), index=False, header=None)

            df = pd.DataFrame(csv_pred_list)
            df.to_csv(os.path.join(output_dir, "valid_ref.csv"), index=False, header=None)

            metrics_dict = compute_metrics(hypothesis=os.path.join(output_dir, "valid_hyp.csv"),
                                           references=[os.path.join(output_dir, "valid_ref.csv")],
                                           no_skipthoughts=True, no_glove=True)

            dev_bleu = round(metrics_dict['Bleu_4'], 3)
            print("  %s = %s " % ("bleu", str(dev_bleu)))
            print("  " + "*" * 20)

            if dev_bleu > best_bleu:
                print("  Best bleu:%s", dev_bleu)
                print("  " + "*" * 20)
                best_bleu = dev_bleu
                # Save best checkpoint for best bleu
                output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir_bleu):
                    os.makedirs(output_dir_bleu)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            return best_bleu, best_loss
        def _streaming_evaluate1(dev_filename, eval_batch_size, output_dir, epoch, global_step, dev_dataset):
            """流式评估 - 内部函数"""
            # 读取评估数据
            eval_examples = read_examples(dev_filename)

            # 创建评估数据集
            eval_features = convert_examples_to_features(eval_examples, self.tokenizer, 
                                                        self.max_source_length, 
                                                        self.max_target_length, stage='dev')

            # 创建张量
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_similarity_ids = torch.tensor([f.similarity_ids for f in eval_features], dtype=torch.long)
            all_similarity_mask = torch.tensor([f.similarity_mask for f in eval_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)

            # 流式评估数据集
            if self.use_ast:
                eval_dataset = StreamASTDataset(
                    all_source_ids, all_source_mask,
                    all_similarity_ids, all_similarity_mask,
                    all_target_ids, all_target_mask,
                    eval_examples,
                    self.ast_processor
                )

                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=ast_collate_fn,
                    num_workers=0
                )
            else:
                eval_data = TensorDataset(
                    all_source_ids, all_source_mask,
                    all_similarity_ids, all_similarity_mask,
                    all_target_ids, all_target_mask
                )

                eval_dataloader = DataLoader(
                    eval_data,
                    batch_size=eval_batch_size,
                    shuffle=False
                )

            print("\n***** 测试一下 *****")
            print("  epoch = %d", epoch)
            print("  Num examples = %d", len(eval_examples))
            print("  Batch size = %d", eval_batch_size)

            # Start Evaling model
            self.model.eval()
            eval_loss, tokens_num = 0, 0
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                if self.use_ast:
                    source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask, source_ast_data = batch
                    source_ids = source_ids.to(self.device)
                    source_mask = source_mask.to(self.device)
                    similarity_ids = similarity_ids.to(self.device)
                    similarity_mask = similarity_mask.to(self.device)
                    target_ids = target_ids.to(self.device)
                    target_mask = target_mask.to(self.device)
                    if source_ast_data:
                        source_ast_data = {k: v.to(self.device) for k, v in source_ast_data.items()}

                    with torch.no_grad():
                        _, loss, num = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids,
                            similarity_mask=similarity_mask, 
                            target_ids=target_ids, 
                            target_mask=target_mask,
                            source_ast_data=source_ast_data
                        )
                else:
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids,
                            similarity_mask=similarity_mask, 
                            target_ids=target_ids, 
                            target_mask=target_mask
                        )

                eval_loss += loss.sum().item()
                tokens_num += num.sum().item()

            # Print loss of dev dataset
            # self.model.train()
            eval_loss = eval_loss / tokens_num
            result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                      'global_step': global_step + 1,
                      'train_loss': round(train_loss, 5)}
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
            print("  " + "*" * 20)

            # save last checkpoint
            # last_output_dir = os.path.join(output_dir, 'checkpoint-last')
            # if not os.path.exists(last_output_dir):
            #     os.makedirs(last_output_dir)
            # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
            # output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            # torch.save(model_to_save.state_dict(), output_model_file)

            # if eval_loss < best_loss:
            #     print("  Best ppl:%s", round(numpy.exp(eval_loss), 5))
            #     print("  " + "*" * 20)
            #     best_loss = eval_loss
            #     # Save best checkpoint for best ppl
            #     output_dir_ppl = os.path.join(output_dir, 'checkpoint-best-ppl')
            #     if not os.path.exists(output_dir_ppl):
            #         os.makedirs(output_dir_ppl)
            #     model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            #     output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
            #     torch.save(model_to_save.state_dict(), output_model_file)

            # Calculate bleu
            eval_examples = read_examples(dev_filename)
            # eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
            eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                         self.max_target_length, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_similarity_ids = torch.tensor([f.similarity_ids for f in eval_features], dtype=torch.long)
            all_similarity_mask = torch.tensor([f.similarity_mask for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_source_ids, all_source_mask, all_similarity_ids, all_similarity_mask)
            dev_dataset['dev_bleu'] = eval_examples, eval_data

            # 如果使用AST，为BLEU计算准备AST数据
            bleu_ast_data = None
            if self.use_ast:
                print("准备BLEU计算AST数据...")
                bleu_source_codes = [ex.source for ex in eval_examples]
                bleu_ast_data = self.ast_processor.prepare_batch_ast_data(bleu_source_codes)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

            self.model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, similarity_ids, similarity_mask = batch

                # 为当前批次准备AST数据
                batch_ast_data = None
                if self.use_ast and bleu_ast_data is not None:
                    # 从批处理AST数据中提取当前批次的AST数据
                    batch_indices = [i for i in range(len(batch[0]))]
                    batch_ast_data = self.extract_batch_ast_data(bleu_ast_data, batch_indices)
                    if batch_ast_data:
                        batch_ast_data = {k: v.to(self.device) for k, v in batch_ast_data.items()}

                with torch.no_grad():
                    if self.use_ast and batch_ast_data is not None:
                        # 修复AST特征维度
                        if batch_ast_data is not None:
                            # 确保AST数据有正确的结构
                            if isinstance(batch_ast_data, dict):
                                # 检查节点数量
                                if 'node_types' in batch_ast_data and batch_ast_data['node_types'].numel() > 0:
                                    # AST数据有效，传递给模型
                                    preds = self.model(
                                        source_ids=source_ids, 
                                        source_mask=source_mask, 
                                        similarity_ids=similarity_ids, 
                                        similarity_mask=similarity_mask,
                                        source_ast_data=batch_ast_data
                                    )
                                else:
                                    print("无效")
                                    # AST数据无效，跳过
                                    preds = self.model(
                                        source_ids=source_ids, 
                                        source_mask=source_mask, 
                                        similarity_ids=similarity_ids, 
                                        similarity_mask=similarity_mask
                                    )
                            else:
                                # batch_ast_data不是字典，跳过
                                preds = self.model(
                                    source_ids=source_ids, 
                                    source_mask=source_mask, 
                                    similarity_ids=similarity_ids, 
                                    similarity_mask=similarity_mask
                                )
                    else:
                        preds = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids, 
                            similarity_mask=similarity_mask
                        )

                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)

            self.model.train()

            csv_pred_list = []
            csv_true_list = []

            for ref, gold in zip(p, eval_examples):
                csv_pred_list.append(gold.target)
                csv_true_list.append(ref)

            df = pd.DataFrame(csv_true_list)
            df.to_csv(os.path.join(output_dir, "ceshi_hyp.csv"), index=False, header=None)

            df = pd.DataFrame(csv_pred_list)
            df.to_csv(os.path.join(output_dir, "ceshi_ref.csv"), index=False, header=None)

            metrics_dict = compute_metrics(hypothesis=os.path.join(output_dir, "ceshi_hyp.csv"),
                                           references=[os.path.join(output_dir, "ceshi_ref.csv")],
                                           no_skipthoughts=True, no_glove=True)

            dev_bleu = round(metrics_dict['Bleu_4'], 3)
            print("  %s = %s " % ("bleu", str(dev_bleu)))
            print("  " + "*" * 20)
            return 1
        for epoch in range(num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                # 根据是否使用AST来解包batch
                if self.use_ast:
                    source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask, source_ast_data = batch
                    # 移动到设备
                    source_ids = source_ids.to(self.device)
                    source_mask = source_mask.to(self.device)
                    similarity_ids = similarity_ids.to(self.device)
                    similarity_mask = similarity_mask.to(self.device)
                    target_ids = target_ids.to(self.device)
                    target_mask = target_mask.to(self.device)
                    if source_ast_data:
                        source_ast_data = {k: v.to(self.device) for k, v in source_ast_data.items()}

                    # 前向传播（使用AST）
                    loss, _, _ = self.model(
                        source_ids=source_ids, 
                        source_mask=source_mask,
                        similarity_ids=similarity_ids,
                        similarity_mask=similarity_mask,
                        target_ids=target_ids,
                        target_mask=target_mask,
                        source_ast_data=source_ast_data
                    )
                else:
                    source_ids, source_mask, similarity_ids, similarity_mask, target_ids, target_mask = batch
                    # 移动到设备
                    source_ids = source_ids.to(self.device)
                    source_mask = source_mask.to(self.device)
                    similarity_ids = similarity_ids.to(self.device)
                    similarity_mask = similarity_mask.to(self.device)
                    target_ids = target_ids.to(self.device)
                    target_mask = target_mask.to(self.device)

                    # 前向传播（不使用AST）
                    loss, _, _ = self.model(
                        source_ids=source_ids,
                        source_mask=source_mask,
                        similarity_ids=similarity_ids,
                        similarity_mask=similarity_mask,
                        target_ids=target_ids,
                        target_mask=target_mask
                    )

                tr_loss += loss.item()
                train_loss = round(tr_loss * gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
        
            if do_eval:
                if epoch >= 15:
                    m = _streaming_evaluate1('../data/second_stage_data/fusion/test.csv', 64, output_dir, 
                                                   epoch, global_step, dev_dataset)
                dev_dataset = {}
                best_bleu, best_loss = _streaming_evaluate(dev_filename, eval_batch_size, output_dir, 
                                                   epoch, global_step, best_bleu, best_loss, dev_dataset)
                


    def batch_ast_data(self, ast_data_list):
        """批量处理AST数据列表"""
        if not ast_data_list or all(d is None for d in ast_data_list):
            return None

        all_node_types = []
        all_node_tokens = []
        all_edge_indices = []
        all_batches = []

        node_offset = 0
        for i, ast_data in enumerate(ast_data_list):
            if ast_data is None:
                # 如果没有AST数据，创建空的
                continue

            num_nodes = len(ast_data['node_types'])

            # 收集节点特征
            all_node_types.append(ast_data['node_types'])
            all_node_tokens.append(ast_data['node_tokens'])

            # 调整边索引
            if ast_data['edge_index'].numel() > 0:
                adjusted_edges = ast_data['edge_index'] + node_offset
                all_edge_indices.append(adjusted_edges)

            # 批索引
            batch_idx = torch.full((num_nodes,), i, dtype=torch.long)
            all_batches.append(batch_idx)

            node_offset += num_nodes

        if not all_node_types:  # 所有AST数据都为空
            return None

        # 合并所有数据
        node_types = torch.cat(all_node_types, dim=0)
        node_tokens = torch.cat(all_node_tokens, dim=0)
        batch = torch.cat(all_batches, dim=0)

        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        return {
            'node_types': node_types,
            'node_tokens': node_tokens,
            'edge_index': edge_index,
            'batch': batch
        }
             
    def test(self, test_filename, test_batch_size, output_dir):
        files = []
        files.append(test_filename)

        for idx, file in enumerate(files):
            print("Test file: {}".format(file))
            eval_examples = read_examples(file)

            eval_features = convert_examples_to_features(
                eval_examples, 
                self.tokenizer, 
                self.max_source_length, 
                self.max_target_length, 
                stage='test'
            )

            # 创建标准张量
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_similarity_ids = torch.tensor([f.similarity_ids for f in eval_features], dtype=torch.long)
            all_similarity_mask = torch.tensor([f.similarity_mask for f in eval_features], dtype=torch.long)

            ast_batch_data = None
            if self.use_ast:
                print("准备测试集AST数据...")
                source_codes = [ex.source for ex in eval_examples]
                ast_batch_data = self.ast_processor.prepare_batch_ast_data(source_codes)

            # 创建数据集
            eval_data = TensorDataset(
                all_source_ids, 
                all_source_mask, 
                all_similarity_ids, 
                all_similarity_mask
            )

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, 
                sampler=eval_sampler, 
                batch_size=test_batch_size
            )

            print("\n***** Running Test *****")
            print(f"  Num examples = {len(eval_examples)}")
            print(f"  Batch size = {test_batch_size}")

            self.model.eval()
            p = []

            # 处理每个批次
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, similarity_ids, similarity_mask = batch

                batch_ast_data = None
                if self.use_ast and ast_batch_data is not None:
                    start_idx = batch_idx * test_batch_size
                    end_idx = min((batch_idx + 1) * test_batch_size, len(eval_examples))
                    batch_indices = list(range(start_idx, end_idx))

                    batch_ast_data = self.extract_batch_ast_data(ast_batch_data, batch_indices)
                    if batch_ast_data:
                        batch_ast_data = {k: v.to(self.device) for k, v in batch_ast_data.items()}

                with torch.no_grad():
                    if self.use_ast and batch_ast_data is not None:
                        preds = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids, 
                            similarity_mask=similarity_mask,
                            source_ast_data=batch_ast_data
                        )
                    else:
                        preds = self.model(
                            source_ids=source_ids, 
                            source_mask=source_mask, 
                            similarity_ids=similarity_ids, 
                            similarity_mask=similarity_mask
                        )

                # 解码
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)

            # ... 保存和评估 ...
            # 以下保持原始代码不变
            csv_pred_list = []
            csv_true_list = []

            for ref, gold in zip(p, eval_examples):
                csv_pred_list.append(gold.target)
                csv_true_list.append(ref)

            df = pd.DataFrame(csv_true_list)
            df.to_csv(os.path.join(output_dir, "my_hyp.csv"), index=False, header=None)

            df = pd.DataFrame(csv_pred_list)
            df.to_csv(os.path.join(output_dir, "my_ref.csv"), index=False, header=None)

            metrics_dict = compute_metrics(hypothesis=os.path.join(output_dir, "my_hyp.csv"),
                                           references=[os.path.join(output_dir, "my_ref.csv")], 
                                           no_skipthoughts=True, 
                                           no_glove=True)
            
    def predict(self, source, similarity):
        encode = self.tokenizer.encode_plus(source, return_tensors="pt", max_length=self.max_source_length, truncation=True, pad_to_max_length=True)
        source_ids = encode['input_ids'].to(self.device)
        source_mask = encode['attention_mask'].to(self.device)
        encode = self.tokenizer.encode_plus(similarity, return_tensors="pt", max_length=self.max_source_length, truncation=True, pad_to_max_length=True)
        similarity_ids = encode['input_ids'].to(self.device)
        similarity_mask = encode['attention_mask'].to(self.device)
        self.model.eval()
        result_list = []
        with torch.no_grad():
            summary_text_ids = self.model(source_ids=source_ids, source_mask=source_mask, similarity_ids=similarity_ids, similarity_mask=similarity_mask)
            for i in range(self.beam_size):
                t = summary_text_ids[0][i].cpu().numpy()
                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                result_list.append(text)
        return result_list
