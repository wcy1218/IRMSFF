import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化模型
model = My_model(codebert_path = '../../model/unixcoder', decoder_layers = 6, fix_encoder = True, beam_size = 10,
                         max_source_length = 64, max_target_length = 64, load_model_path = '../pretrained_model/first_stage/checkpoint-best-bleu/pytorch_model.bin', l2_norm=True, fusion=True, use_ast=True, gnn_type='gcn')

# 模型训练
model.train(train_filename ='../data/second_stage_data/fusion/train.csv', train_batch_size = 32, num_train_epochs = 120, learning_rate = 5e-5,
            do_eval = True, dev_filename ='../data/second_stage_data/fusion/valid.csv', eval_batch_size = 32, output_dir ='../pretrained_model/second_stage')

# 加载微调过的模型
model = My_model(codebert_path = '../../model/unixcoder', decoder_layers = 6, fix_encoder = True, beam_size = 10,
                         max_source_length = 64, max_target_length = 64, load_model_path = '../pretrained_model/second_stage/checkpoint-best-bleu/pytorch_model.bin', l2_norm=True, fusion=True, use_ast=True, gnn_type='gcn')

# 模型测试
model.test(test_filename ='../data/second_stage_data/fusion/test.csv', test_batch_size = 16, output_dir ='../result/final')
