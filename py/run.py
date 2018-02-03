# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import math
import data_util
import numpy as np
import tensorflow as tf
from seqModel import SeqModel
from data_iterator import DataIterator
from tensorflow.python.client import timeline


# mode
tf.app.flags.DEFINE_string("mode", "TRAIN", "TRAIN|FORCE_DECODE|BEAM_DECODE|DUMP_LSTM")

# datasets, paths, and preprocessing
tf.app.flags.DEFINE_string("model_dir", "./model", "model_dir/data_cache/n model_dir/saved_model; model_dir/log.txt .")
tf.app.flags.DEFINE_string("train_path_from", "./train", "the absolute path of raw source train file.")
tf.app.flags.DEFINE_string("dev_path_from", "./dev", "the absolute path of raw source dev file.")
tf.app.flags.DEFINE_string("test_path_from", "./test", "the absolute path of raw source test file.")

tf.app.flags.DEFINE_string("train_path_to", "./train", "the absolute path of raw target train file.")
tf.app.flags.DEFINE_string("dev_path_to", "./dev", "the absolute path of raw target dev file.")
tf.app.flags.DEFINE_string("test_path_to", "./test", "the absolute path of raw target test file.")

tf.app.flags.DEFINE_string("decode_output", "./output", "beam search decode output.")

tf.app.flags.DEFINE_string("force_decode_output", "force_decode.txt", "the file name of the score file as the output of force_decode. The file will be put at model_dir/force_decode_output")
tf.app.flags.DEFINE_string("dump_lstm_output", "dump_lstm.pb", "the file to save hidden states as a protobuffer as the output of dump_lstm. The file will be put at model_dir/dump_lstm_output")

# tuning hypers
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.8,"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,"Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training/evaluation.")
tf.app.flags.DEFINE_integer("from_vocab_size", 10000, "from vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 10000, "to vocabulary size.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("n_epoch", 500,"Maximum number of epochs in training.")
tf.app.flags.DEFINE_integer("L", 30, "max length")
tf.app.flags.DEFINE_integer("n_bucket", 10,"num of buckets to run.")
tf.app.flags.DEFINE_integer("patience", 10, "exit if the model can't improve for $patence evals")

# devices
tf.app.flags.DEFINE_string("N", "000", "GPU layer distribution: [input_embedding, lstm, output_embedding]")

# training parameter
tf.app.flags.DEFINE_boolean("withAdagrad", True,"withAdagrad.")
tf.app.flags.DEFINE_boolean("fromScratch", True,"withAdagrad.")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,"save Model at each checkpoint.")
tf.app.flags.DEFINE_boolean("profile", False, "False = no profile, True = profile")

# for beam_decode
tf.app.flags.DEFINE_integer("beam_size", 10,"the beam size")
tf.app.flags.DEFINE_boolean("print_beam", False, "to print beam info")
tf.app.flags.DEFINE_float("min_ratio", 0.5, "min_ratio.")
tf.app.flags.DEFINE_float("max_ratio", 1.5, "max_ratio.")

# GPU configuration
tf.app.flags.DEFINE_boolean("allow_growth", False, "allow growth")

# With Attention
tf.app.flags.DEFINE_boolean("attention", False, "with_attention")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(10, 10), (22, 22)]
_beam_buckets = [10, 22]

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()][::-1]   #输入反过来
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_util.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

#读入test的数据，并将输入倒序，记录下输入句子的bucket以及对应的位置。
def read_data_test(source_path):
    order = []
    data_set = [[] for _ in _beam_buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source:
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()][::-1]# 将输入倒序
            for bucket_id, source_size in enumerate(_beam_buckets):
                if len(source_ids) < source_size:
                    order.append((bucket_id, len(data_set[bucket_id])))
                    data_set[bucket_id].append(source_ids)
                    break
            source = source_file.readline()
    return data_set, order

def create_model(session, run_options, run_metadata):
    devices = get_device_address(FLAGS.N)
    dtype = tf.float32
    model = SeqModel(FLAGS._buckets,
                     FLAGS.size,
                     FLAGS.real_vocab_size_from,
                     FLAGS.real_vocab_size_to,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     withAdagrad = FLAGS.withAdagrad,
                     dropoutRate = FLAGS.keep_prob,
                     dtype = dtype,
                     devices = devices,
                     topk_n = FLAGS.beam_size,
                     run_options = run_options,
                     run_metadata = run_metadata,
                     with_attention = FLAGS.attention,
                     beam_search = FLAGS.beam_search,
                     beam_buckets = _beam_buckets
                     )
    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)
    if FLAGS.mode == "DUMP_LSTM" or FLAGS.mode == "BEAM_DECODE" or FLAGS.mode == 'FORCE_DECODE' or (not FLAGS.fromScratch) and ckpt:
        mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.variables_initializer(model.beam_search_vars))
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def get_device_address(s):
    add = []
    if s == "":
        for i in range(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]

    return add

def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name)

def train():
    #1.读入train数据和dev数据
    mylog_section('READ DATA')
    from_train = None
    to_train = None
    from_dev = None
    to_dev = None

    from_train, to_train, from_dev, to_dev, _, _ = data_util.prepare_data(
        FLAGS.data_cache_dir,
        FLAGS.train_path_from,
        FLAGS.train_path_to,
        FLAGS.dev_path_from,
        FLAGS.dev_path_to,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)

    train_data_bucket = read_data(from_train, to_train)
    dev_data_bucket = read_data(from_dev, to_dev)
    _, _, real_vocab_size_from, real_vocab_size_to = data_util.get_vocab_info(FLAGS.data_cache_dir)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    # train_n_tokens = total training target size
    train_n_tokens = np.sum([np.sum([len(items[1]) for items in x]) for x in train_data_bucket])
    train_bucket_sizes = [len(train_data_bucket[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    dev_bucket_sizes = [len(dev_data_bucket[b]) for b in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))

    mylog_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_dev = int(dev_total_size / batch_size)
    steps_per_checkpoint = int(steps_per_epoch / 2)
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_buckets: {}".format(FLAGS._buckets))
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("bucket sizes: {}".format(train_bucket_sizes))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("bucket sizes: {}".format(dev_bucket_sizes))
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))


    mylog_section("IN TENSORFLOW")

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    with tf.Session(config=config) as sess:
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL/SUMMARY/WRITER")
        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, run_options, run_metadata)

        mylog_section("All Variables")
        show_all_variables()

        # Data Iterators
        mylog_section("Data Iterators")
        dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale)
        iteType = 0
        if iteType == 0:
            mylog("Itetype: withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            mylog("Itetype: withSequence")
            ite = dite.next_sequence()

        # statistics during training
        step_time, loss = 0.0, 0.0
        current_step = 0
        low_ppx = float("inf")
        steps_per_report = 30
        n_targets_report = 0
        report_time = 0
        n_valid_sents = 0
        n_valid_words = 0
        patience = FLAGS.patience

        mylog_section("TRAIN")
        while current_step < total_steps:
            # start
            start_time = time.time()
            # data and train
            source_inputs, target_inputs, target_outputs, target_weights, bucket_id = ite.next()
            L = model.step(sess, source_inputs, target_inputs, target_outputs, target_weights, bucket_id)
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += L
            current_step += 1
            # 此处 weights 等数据的格式是 len(weights) == 句子长度
            # len(weights[0]) 是 batch size
            n_valid_sents += np.sum(np.sign(target_weights[0]))
            n_valid_words += np.sum(target_weights)
            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(target_weights)

            #显示信息
            if current_step % steps_per_report == 0:
                sect_name = "STEP {}".format(current_step)
                msg = "StepTime: {:.2f} sec Speed: {:.2f} targets/s Total_targets: {}".format(
                    report_time / steps_per_report, n_targets_report * 1.0 / report_time, train_n_tokens)
                mylog_line(sect_name, msg)
                report_time = 0
                n_targets_report = 0

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()

            #达到半个epoch，计算ppx(dev)
            if current_step % steps_per_checkpoint == 0:
                i_checkpoint = int(current_step / steps_per_checkpoint)
                # train_ppx
                loss = loss / n_valid_words
                train_ppx = math.exp(float(loss)) if loss < 300 else float("inf")
                learning_rate = model.learning_rate.eval()

                # dev_ppx
                dev_loss, dev_ppx = evaluate(sess, model, dev_data_bucket)

                # report
                sect_name = "CHECKPOINT {} STEP {}".format(i_checkpoint, current_step)
                msg = "Learning_rate: {:.4f} Dev_ppx: {:.2f} Train_ppx: {:.2f}".format(learning_rate, dev_ppx,train_ppx)
                mylog_line(sect_name, msg)

                # save model per checkpoint
                if FLAGS.saveCheckpoint:
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph=False)
                    msg = "Model saved using {:.2f} sec at {}".format(time.time() - s, checkpoint_path)
                    mylog_line(sect_name, msg)

                # save best model
                if dev_ppx < low_ppx:
                    patience = FLAGS.patience
                    low_ppx = dev_ppx
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "best")
                    s = time.time()
                    model.best_saver.save(sess, checkpoint_path, global_step=0, write_meta_graph=False)
                    msg = "Model saved using {:.2f} sec at {}".format(time.time() - s, checkpoint_path)
                    mylog_line(sect_name, msg)
                else:
                    patience -= 1
                    #每次当 dev_ppx >= low_ppx时 学习步长减半
                    sess.run(model.learning_rate_decay_op)
                    msg = 'dev_ppx:{}, low_ppx:{}'.format(str(dev_ppx), str(low_ppx))
                    mylog_line(sect_name, msg)
                    msg = 'dev_ppx >= low_ppx，patience ={}, learning_reate ={}'.format(str(patience), str(model.learning_rate.eval()))
                    mylog_line(sect_name, msg)

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents, n_valid_words = 0.0, 0.0, 0, 0


#达到半个epoch，计算ppx(dev)
def evaluate(sess, model, data_set):
    # Run evals on development set and print their perplexity/loss.
    sess.run(model.dropout10_op)# 验证的时候dropout设置为1，也就是不dropout
    loss = 0.0
    n_steps = 0
    n_valids = 0
    batch_size = FLAGS.batch_size

    dite = DataIterator(model, data_set, len(FLAGS._buckets), batch_size, None)
    ite = dite.next_sequence(stop=True)
    for sources, inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, sources, inputs, outputs, weights, bucket_id, forward_only=True)
        loss += L
        n_steps += 1
        n_valids += np.sum(weights)
    loss = loss / (n_valids)
    ppx = math.exp(loss) if loss < 300 else float("inf")
    sess.run(model.dropoutAssign_op)  #验证结束需要将 dropout恢复原来的设置。
    return loss, ppx

def beam_decode():
    mylog("Reading Data...")
    from_test = None
    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_util.get_vocab_info(FLAGS.data_cache_dir)

    FLAGS._buckets = _buckets
    FLAGS._beam_buckets = _beam_buckets
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    # 得到test文件转换成ids的地址。
    from_test = data_util.prepare_test_data(FLAGS.data_cache_dir, FLAGS.test_path_from, from_vocab_path)

    test_data_bucket, test_data_order = read_data_test(from_test)

    test_bucket_sizes = [len(test_data_bucket[b]) for b in xrange(len(_beam_buckets))]
    test_total_size = int(sum(test_bucket_sizes))
    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("_beam_buckets: {}".format(FLAGS._beam_buckets))
    mylog("BEAM_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    with tf.Session(config=config) as sess:
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata)
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))
        batch_size = FLAGS.batch_size

        dite = DataIterator(model, test_data_bucket, len(_beam_buckets), batch_size, None, data_order=test_data_order)
        ite = dite.next_original()
        i_sent = 0
        targets = []

        for source_inputs, bucket_id, length in ite:
            print("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
            i_sent += 1

            results = []  # (sentence,score)
            scores = [0.0] * FLAGS.beam_size
            sentences = [[] for x in xrange(FLAGS.beam_size)]
            beam_parent = range(FLAGS.beam_size)

            target_inputs = [data_util.GO_ID] * FLAGS.beam_size
            min_target_length = int(length * FLAGS.min_ratio) + 1
            max_target_length = int(length * FLAGS.max_ratio) + 1  # include EOS
            for i in xrange(max_target_length):
                if i == 0:
                    top_value, top_index, eos_value = model.beam_step(sess, bucket_id, index=i, sources=source_inputs, target_inputs=target_inputs)
                else:
                    top_value, top_index, eos_value = model.beam_step(sess, bucket_id, index=i, target_inputs=target_inputs, beam_parent=beam_parent)
                # expand
                global_queue = [] #没预测一个词之前都重新定义，用来记录加入句子以后的分数以及对应的句子，最后根据分数排名选出最佳的句子。
                if i == 0:  #如果是decoder的第一步，则只取第一行作为输出，作为第二次的输入。
                    nrow = 1
                else:
                    nrow = FLAGS.beam_size

                if i == max_target_length - 1:  # last_step
                    for row in xrange(nrow):
                        score = scores[row] + np.log(eos_value[0][row, 0])
                        word_index = data_util.EOS_ID
                        beam_index = row
                        global_queue.append((score, beam_index, word_index))
                else:
                    for row in xrange(nrow):  # 对每一个parent的子预测结果进行预测，xrange(nrow)就是循环遍历每一个Parent。
                        for col in xrange(top_index[0].shape[1]):  #对每一个parent 的 top_index的每一个预测结果进行计算。 top_index的每一列就是一个预测结果。
                            score = scores[row] + np.log(top_value[0][row, col]) #新的分数是原parent的句子的分数*后面生成的单词的分数。
                            word_index = top_index[0][row, col]
                            beam_index = row  #parent

                            global_queue.append((score, beam_index, word_index))
                global_queue = sorted(global_queue, key=lambda x: -x[0])
                if FLAGS.print_beam:
                    print("--------- Step {} --------".format(i))
                target_inputs = []
                beam_parent = []
                scores = []
                temp_sentences = []
                #对排序好的global_queue取前beam_size个存入target_inputs、beam_parent、scores、temp_sentences中供下一步预测使用。
                for j, (score, beam_index, word_index) in enumerate(global_queue):
                    if word_index == data_util.EOS_ID:
                        if len(sentences[beam_index]) + 1 < min_target_length:
                            continue
                        results.append((sentences[beam_index] + [word_index], score)) #每预测一个句子，就加入到results中。
                        if FLAGS.print_beam:
                            print("*Beam:{} Father:{} word:{} score:{}".format(j, beam_index, word_index, score))
                        continue
                    if FLAGS.print_beam:
                        print("Beam:{} Father:{} word:{} score:{}".format(j, beam_index, word_index, score))
                    beam_parent.append(beam_index)
                    target_inputs.append(word_index)
                    scores.append(score)
                    temp_sentences.append(sentences[beam_index] + [word_index])
                    if len(scores) >= FLAGS.beam_size:  #选取前beam_size个结果保存供下次使用。
                        break
                # can not fill beam_size, just repeat the last one，不足beam_size个数据，用最后一个数据填充。
                while len(scores) < FLAGS.beam_size and i < max_target_length - 1:
                    beam_parent.append(beam_parent[-1])
                    target_inputs.append(target_inputs[-1])
                    scores.append(scores[-1])
                    temp_sentences.append(temp_sentences[-1])
                sentences = temp_sentences
            # print the 1 best
            #将一个source的所有预测的句子排序
            results = sorted(results, key=lambda x: -x[1])
            #选取最好的结果加入到targets中。
            targets.append(results[0][0])
        #对所有的预测的句子转换成word并写入文件中。
        data_util.ids_to_tokens(targets, to_vocab_path, FLAGS.decode_output)


def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)

def mylog_section(section_name):
    mylog("======== {} ========".format(section_name))

def log_flags():
    members = FLAGS.__dict__['__flags'].keys()
    mylog_section("FLAGS")
    for attr in members:
        mylog("{}={}".format(attr, getattr(FLAGS, attr)))

def mylog_line(section_name, message):
    mylog("[{}] {}".format(section_name, message))

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def parsing_flags():
    FLAGS.data_cache_dir = os.path.join(FLAGS.model_dir,'data_cache')
    FLAGS.saved_model_dir = os.path.join(FLAGS.model_dir, "saved_model")
    FLAGS.summary_dir = FLAGS.saved_model_dir

    mkdir(FLAGS.model_dir)
    mkdir(FLAGS.data_cache_dir)
    mkdir(FLAGS.saved_model_dir)
    mkdir(FLAGS.summary_dir)

    # for logs
    log_path = os.path.join(FLAGS.model_dir, "log.{}.txt".format(FLAGS.mode))  #log地址
    filemode = 'w' if FLAGS.fromScratch else "a"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode=filemode)

    FLAGS.beam_search = False

    log_flags()

def main():
    #读取超参数
    parsing_flags()

    if FLAGS.mode == 'TRAIN':
        train()

    if FLAGS.mode == 'FORCE_DECODE':
        mylog(
            "\nWARNING: \n 1. The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n 2. The score is -sum(log(p)) with base e and includes EOS. \n")

        FLAGS.batch_size = 1
        FLAGS.score_file = os.path.join(FLAGS.model_dir, FLAGS.force_decode_output)  #预测结果的输出文件地址

    if FLAGS.mode == "BEAM_DECODE":
        FLAGS.batch_size = FLAGS.beam_size
        FLAGS.beam_search = True
        beam_decode()



if __name__ == '__main__':
    main()