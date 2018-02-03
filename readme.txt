基于Seq2seq实现的是一个简单的复制操作，输出对输入复制。  数据是自己造的。
配置了DropOut、reverse、attention、feed_input、beam_search方法。
相关配置可以在sh文件中修改。



1.SeqModel.py
    (1)def init_beam_decoder():
         画额外的图，定义两个variables: before_state, after_state
         定义options: after2before_ops,用于根据beam_parents更新下一步输入的ht、ct。
         调用beam_with_buckets()，根据不同输入的bucket画不同的图
    (2)def beam_with_buckets():
         根据不同的输入的bucket，输入不同长度的source_inputs, sources[:source_length],target的输入一直都是第一个输入，inputs[:1]
         调用beam_basic_seq2seq得到：
         _hts, encoder2beforestate, decoder2afterstate.
         对_hts用softmax处理，提取topk，得到：
         topk_value,topk_index,eos_value
    (3)def beam_basic_seq2seq():
         画seq2seq的图
         定义encoder、decoder、encoder2before_state_ops、decoder2after_state_ops
         返回 decoder_outputs, encoder2before_state_ops, decoder2after_state_ops
    (4)def beam_step():
         输入实际的Input，output。调用session_run执行。


2.run.py:
    (1)def beam_decode():
         图以外的操作，真正的beam_search逻辑计算。通过调用model.beam_step 得到 top_value,top_index,eos_value,生成新的每一步的词的概率，根据beam_search的思想选取合适的词做为输出。
         1.读入数据。
         2.创建模型，此处为恢复训练模型参数。
         3.show_all_variables()
         4.设置dropoutRate 为1
         5.循环调取每一个test数据：
               （1）设置5个中间存储数据 results、scores、sentences、beam_parent、target_inputs
               （2）循环每一个生成的词，一直到最长句子的限制，for i in xrange(max_target_length):
                        初始化global_queue
                        1.如果是第一步decoder预测，调用model.beam_step，输入是sources 和 target_inputs,得到top_value,top_index,eos_value
                          如果不是第一步decoder预测，调用model.beam_step,输入是target_inputs, beam_paren 。 target_inputs是由上面的预测top_index以及下面的beam_search计算得到的。
                        2.如果是decoder的第一步，则只取第一行作为输出，作为第二次的输入。nrow = 1
                          否则 nrow = FLAGS.beam_size
                        3.如果预测步数到了max_target_length，则只看eos的分数，将其加到句子的总分数，在加入global_queue中。
                          否则：
                           对每一个parent的子预测结果进行预测，xrange(nrow)就是循环遍历每一个Parent。
                               对每一个parent 的 top_index的每一个预测结果进行计算。 top_index的每一列就是一个预测结果。
                                   新的分数是原parent的句子的分数*后面生成的单词的分数。
                                   global_queue.append((score, beam_index, word_index))
                        4.对global_queue按照分数排序。
                        5.对排序好的global_queue取前beam_size个存入target_inputs、beam_parent、scores、temp_sentences中供下一步预测使用。
                        6.不足beam_size个数据，用最后一个数据填充。
                （3）将一个source的所有预测的句子排序，选取最好的结果加入到targets中。
         6.对所有的预测的句子转换成word并写入文件中。


加入attention:
注意：before_state和 before_h_att 的区别是 不加 attention的输入有h 和 c， 加入attention的只有h~。
1.def init_beam_decoder()
    (1)定义before_h_att、 after_h_att、 top_states_transform_4s、 top_states_4s、   hatt_after2before_ops
    (2)图hatt_after2before_ops
    (3)调用beam_with_buckets（）函数，根据输入的不同长度画不同的图。
    (3)定义hatt2a_ops、top_states_transform_4_ops、top_states_4_ops
    (4)调用beam_attention_seq2seq（），画具体的seq2seq的图，生成 _hts, _, e2b, d2a, hatt2a, top_states_transform_4_op, top_states_4_op
    (5)对输出的_hts做softmax处理，得到topk_value、topk_index、eos_value
    (6)beam_attention_seq2seq（）:
            1.定义相关参数。
            2.encoder：
                (1)encoder_outputs, encoder_state
                (2)encoder2before_ops、top_states_transform_4_op、top_states_4_op
            3.定义get_context(query)，计算C每次流程都一样，只是ht不一样，所以可以写成这个通用的函数，再用生成的C与Ht得到ht~。
            4.decoder:
                (1)输出：decoder_outputs = [h_att]
            5.decoder2after_ops
            6.hatt2after_ops




























