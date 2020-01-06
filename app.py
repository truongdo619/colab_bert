import os
from spellCorrector import SpellCorrector
from model import Model
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow as tf
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with open('counts_1grams.txt') as fopen:
    f = fopen.read().split("\n")[:-1]

words=set()
for line in f:
    words.add(line)

corrector = SpellCorrector(words)
possible_states = corrector.edit_candidates('tục')

print(possible_states)



def tokens_to_masked_ids(tokens, mask_ind):
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = "[MASK]"
    masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids

BERT_VOCAB = './uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = './uncased_L-12_H-768_A-12/bert_model.ckpt'

tokenization.validate_case_matches_checkpoint(True, BERT_INIT_CHKPNT)
tokenizer=tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)

text = 'tỉnh Đồng Nai tiếp tục đồng loạt ra quân kiểm tra'
text_mask = text.replace('tục', '**mask**')


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()

print(model.X)
sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')
cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'cls')

saver = tf.train.Saver(var_list = var_lists + cls)
saver.restore(sess, BERT_INIT_CHKPNT)
replaced_masks = [text_mask.replace('**mask**', state) for state in possible_states]

def generate_ids(mask):
    tokens = tokenizer.tokenize(mask)
    input_ids = [tokens_to_masked_ids(tokens, i) for i in range(len(tokens))]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, tokens_ids

ids = [generate_ids(mask) for mask in replaced_masks]
tokens, input_ids, tokens_ids = list(zip(*ids))
indices, ids = [], []
for i in range(len(input_ids)):
    indices.extend([i] * len(input_ids[i]))
    ids.extend(input_ids[i])

masked_padded = tf.keras.preprocessing.sequence.pad_sequences(ids,padding='post')

preds = sess.run(tf.nn.log_softmax(model.logits), feed_dict = {model.X: masked_padded})

indices = np.array(indices)
scores = []

for i in range(len(tokens)):
    filter_preds = preds[indices == i]
    total = np.sum([filter_preds[k, k + 1, x] for k, x in enumerate(tokens_ids[i])])
    scores.append(total)

prob_scores = np.array(scores) / np.sum(scores)
probs = list(zip(possible_states, prob_scores))
probs.sort(key = lambda x: x[1])

print(probs)
