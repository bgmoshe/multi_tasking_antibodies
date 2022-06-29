with open(__file__, "r") as fh:
  code = str(fh.readlines())

"""# Imports & Installs"""

from Bio import SeqIO, Seq
import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing

#from matplotlib import pyplot as plt
from functools import lru_cache
import scipy
import scipy.stats
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
import keras

np.seterr(all = 'raise')

def set_seed(seed):
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


"""# Models testing and training

## Dataframe creations
"""

def df_to_antibody(df):
  avg_df = df[df.library == "average"]
  antibody_columns = df.selection.values
  d = { antibody+"_escape": list() for antibody in antibody_columns}
  d['site'] = list()
  d['mutation'] = list()
  for key, grp_df in avg_df.groupby(["site", "mutation"]):
    site, mut = key
    d['site'].append(site)
    d['mutation'].append(mut)
    s = set(antibody_columns)
    for row_name, row in grp_df.iterrows():
      sel = row['selection']
      escape = row['mut_escape_frac_single_mut']
      d[f"{sel}_escape"].append(escape)
      s.remove(sel)
    if len(s) > 0:
      for sel in s:
        d[f"{sel}_escape"].append(np.nan)
  return pd.DataFrame.from_dict(d)

escape_fracs_df = pd.read_csv(r"../data/original/escape_fracs.csv")

selections_columns = [f"{sel}_escape" for sel in set(escape_fracs_df.selection.values)]
escape_with_all_selections = df_to_antibody(escape_fracs_df)

covid19_selections_columns = ['COV2-2050_400_escape',
                              'COV2-2832_400_escape',
                              'COV2-2165_400_escape',
                              'COV2-2479_400_escape',
                              'COV2-2096_400_escape',
                              'COV2-2499_400_escape',
                              'COV2-2082_400_escape',
                              'COV2-2677_400_escape',
                              'COV2-2094_400_escape',
                              #'CR3022_400_escape'
                              # The cocktails dataset
                              #'LY-CoV016',
                              #'REGN10933',
                              #'REGN10987',
                              ]

#escape_with_all_selections.head()

bind_df = pd.read_csv(r"../data/original/single_mut_effects.csv")
bind_df = bind_df.drop(columns=["site_SARS2", "wildtype", "mutation", "mutation_RBD"], axis=1)
bind_df = bind_df.rename({"site_RBD":"site",
                          "mutant":"mutation"}, axis=1)
#bind_df.head()

bind_and_escapes_df = bind_df.merge(escape_with_all_selections, how='inner', on=["site", "mutation"])
bind_and_escapes_df = bind_and_escapes_df.drop(labels=["bind_lib1",
                                                       "bind_lib2",
                                                       "expr_lib1",
                                                       "expr_lib2"],
                                               axis=1)

"""## Df to Numpy"""

@lru_cache()
def get_vocab():
    amino_acids = ['A', 'C', 'D', 'E', 'F',
                   'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R',
                   'S', 'T', 'V', 'W', 'Y'
                  ]
    #amino_acids = [
    #    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
    #   'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
    #    'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    #]
    vocabulary = { aa: idx for idx, aa in enumerate(sorted(amino_acids)) }
    #vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(amino_acids)) }
    return vocabulary

def get_data_w_bind(df, 
                    selection,
                    binding,
                    region=(330, 530),
                    with_transform=True, 
                    full_seq=False,
                    dropna=False,
                    test_size=0.1,
                    seed=None):
  if isinstance(selection, str):
    selection = [selection]
  if isinstance(binding, str):
    binding = [binding]
  if binding is None:
    binding = []
  if dropna:
    df = df.dropna()

  if full_seq:
    def _featurize(orig_seq):
      vocabulary = get_vocab()
      seq = np.array([vocabulary[word] for word in orig_seq])
      return seq
    mut_and_site = df[['site', 'mutation']]
    print(mut_and_site.head())
    seq_wt = str(SeqIO.read(r'../data/original/cov2_spike_wt.fasta', 'fasta').seq)
    seq_wt = seq_wt[region[0]: region[1]+1]
    seq_wt = _featurize(seq_wt)
    vocabulary = get_vocab()
    x_s = np.array([np.concatenate([seq_wt[:ind-1], 
                                    [vocabulary[mut]], 
                                    seq_wt[ind:]]) 
                    for ind, mut in mut_and_site.values])
  else:
    df_featurized = df[['site', 'mutation']]
    featurized_mutants = df_featurized["mutation"].apply(lambda _: get_vocab()[_])
    x_s = np.hstack((df_featurized[['site']].to_numpy(), featurized_mutants.to_numpy().reshape(-1, 1)))

  y_selection = df[selection].to_numpy()
  y_binding = df[binding].to_numpy()
  y_s = np.concatenate((y_selection,y_binding), axis=1)
  #print("zzz", y_selection.shape)
  threshold = 10*(np.quantile(a=y_selection, q=0.5, axis=0))  
  #print("zzz", threshold)
  if y_s.shape[-1] == 1:
    stratify = (y_selection >= threshold).any(axis=1).squeeze()
  else:
    stratify = None
  X_train, X_test, y_train, y_test = model_selection.train_test_split(x_s,
                                                                      y_s, 
                                                                      test_size = test_size,
                                                                      stratify=stratify,
                                                                      random_state=seed
                                                                      )

  y_train =  y_train.reshape((len(y_train), -1, 1))
  y_test = y_test.reshape((len(y_test), -1, 1))


  if not full_seq:
    X_train = [X_train[:,0], X_train[:, 1]]
    X_test = [X_test[:,0], X_test[:, 1]]
  
  cols_to_index = {
      col: df[selection+binding].columns.get_loc(col) for col in df[selection+binding].columns.to_list()
  }

  return X_train, y_train, X_test, y_test, cols_to_index

def featurize(orig_seq):
      vocabulary = get_vocab()
      seq = np.array([vocabulary[word] for word in orig_seq])
      return seq
      
def get_data_w_bind_no_split(df, 
                             selection,
                             binding,
                             region=(330, 530),
                             full_seq=False,
                             dropna=False):
  if isinstance(selection, str):
    selection = [selection]
  if isinstance(binding, str):
    binding = [binding]
  if binding is None:
    binding = []
  if dropna:
    df = df.dropna()

  if full_seq:
    def _featurize(orig_seq):
      vocabulary = get_vocab()
      seq = np.array([vocabulary[word] for word in orig_seq])
      return seq
    mut_and_site = df[['site', 'mutation']]
    seq_wt = str(SeqIO.read('cov2_spike_wt.fasta', 'fasta').seq)
    seq_wt = seq_wt[region[0]: region[1]+1]
    seq_wt = _featurize(seq_wt)
    vocabulary = get_vocab()
    x_s = np.array([np.concatenate([seq_wt[:ind-1], 
                                    [vocabulary[mut]], 
                                    seq_wt[ind:]]) 
                    for ind, mut in mut_and_site.values])
  else:
    df_featurized = df[['site', 'mutation']]
    featurized_mutants = df_featurized["mutation"].apply(lambda _: get_vocab()[_])
    x_s = np.hstack((df_featurized[['site']].to_numpy(), featurized_mutants.to_numpy().reshape(-1, 1)))
  
  y_selection = df[selection].to_numpy()
  y_binding = df[binding].to_numpy()
  y_s = np.concatenate((y_selection,y_binding), axis=1)


  if not full_seq:
    X_s = [X[:,0], X[:, 1]]
  
  cols_to_index = {
      col: df[selection+binding].columns.get_loc(col) for col in df[selection+binding].columns.to_list()
  }

  return x_s, y_s, cols_to_index


"""# Models part

## Helper funcs
"""

from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K


def escape_and_bind_model(hidden_dims,
                          max_seq_len=201,
                          embed_mutation_dim=40,#8,#40,#100,
                          prob_dims=1,
                          regression_dim=1,
                          uncertanity_layer = False,
                          batch_norm=None, rate=None):
  seq_input = keras.Input(shape=(max_seq_len, ))
  embed_seq = keras.layers.Embedding(input_dim=len(get_vocab()),#+1, 
                                     output_dim=embed_mutation_dim, 
                                     name='embed_mutation',
                                     #embeddings_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.1),
                                     input_length=max_seq_len)(seq_input)
  #embed_seq = tf.keras.layers.CategoryEncoding(num_token=len(get_vocab()))(seq_input)
  
  embed_seq = keras.layers.Flatten()(embed_seq)
  curr = embed_seq
  #curr = keras.layers.ReLU()(embed_seq)
  for i, hidden_size in enumerate(hidden_dims):
      dense = keras.layers.Dense(hidden_size, activation='relu', name=f"{i+1}_th_dense",
                                 #kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.01)
                                 #kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.1)
                                 )
      curr = dense(curr)
      if batch_norm:
          curr = keras.layers.BatchNormalization()(curr)
      if rate:
          curr = keras.layers.Dropout(rate)(curr)
          
  outputs = [keras.layers.Dense(1, 
                                activation="sigmoid",
                                name=f"prob_output_{i}",
                                #kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.1)
                                )(curr)
                                 for i in range(1, prob_dims+1)] + \
            [keras.layers.Dense(1, name=f"reg_output_{i}")(curr)  for i in range(1, regression_dim+1) ]
            
  #outputs = [keras.layers.Dense(1, activation="sigmoid", name=f"prob_output_{i}", kernel_initializer=tf.initializers.constant(0.001))(curr) for i in range(1, prob_dims+1)] + \
  #          [keras.layers.Dense(1, name=f"reg_output_{i}")(curr)  for i in range(1, regression_dim+1) ]
  
  #outputs = [keras.layers.Dense(1, activation="sigmoid", name=f"prob_output_{i}")(keras.layers.Flatten()(curr)) for i in range(1, prob_dims+1)] + \
  #              [keras.layers.Dense(1, name=f"reg_output_{i}")(curr)  for i in range(1, regression_dim+1) ]
  
  return keras.Model(inputs=seq_input,
                     outputs=outputs)

def get_experiment_data_w_bind(num_of_iterations,
                               df,
                               columns_subsets_groups,
                               binding_columns,
                               epochs,
                               verbose=1,
                               batch_size=32,
                               lr = 1e-3,
                               confidence_bound=0.95,
                               binarize = None,
                               loss_weights=None,
                               return_models = False,
                               hidden_layers=None,
                               return_val_data=False,
                               qs=None,
                               with_stop_call=False,
                               uncertanity_layer = False,
                               auto_weight=False,
                               test_size=0.3,
                               with_aug=False,
                               with_sample_weight=False,                               
                               dropout=None,
                               normalized=False,
                               downsample=False,
                               ros=False,
                               temp=None,
                               refined_bin=False,
                               save_embeddings=False
                               ):
  from scipy.special import softmax
  meta = str(locals())
  print(f"params: {locals()}")
  hists = []
  full_history = [[] for _ in columns_subsets_groups]
  final_models = list()
  if hidden_layers is None:
    hidden_layers = []#[128]
  all_cols_names = list(set().union(*columns_subsets_groups))
  vals = list()
  trains = list()
  threshold = 10*(df[covid19_selections_columns].quantile(0.5))
  if loss_weights is None:
    total_len = len(all_cols_names)+len(binding_columns)
    loss_weights = np.ones(total_len)/total_len#[1 for _ in range(len(all_cols_names)+len(binding_columns))]
      #if len(binding_columns) > 0:
      #  loss_weights = [20 for _ in range(len(cols))] +  [1 for _ in range(len(binding_columns))]
      #else:
      #  loss_weights = [1 for _ in range(len(cols))]

    print(f"Loss Weights = {loss_weights}")
    print(loss_weights)
  
  seq_wt = str(SeqIO.read('cov2_spike_wt.fasta', 'fasta').seq)
  variants = [[(501, "Y")],  # Alpha
              [(484, "K"), (501, "Y")],  # Alpha with E484K
              [(417, "N"), (484, "K"), (501, "Y")],  # Beta
              [(417, "T"), (484, "K"), (501, "Y")],  # Gamma
              [(452, "R"), (478, "K")],  # Delta
              [(417, "N"), (452, "R"), (478, "K")],  # Delta Plus
              [(339, "D"), (371, "L"), (373, "P"), (375, "F"),
               (417, "N"), (440, "K"), (446, "S"), (477, "N"),
               (478, "K"), (484, "A"), (493, "R"), (496, "S"),
               (498, "R"), (501, "Y"), (505, "H"), (547, "K")],  # Omicron
              ]  
                 
  def _other_variants_with_char(seq, variants):
    new_variant = []
    for loc, ch in variants:
      curr = get_vocab()[seq[loc-1]]
      ch_res=None
      #print(loc, ch, curr, seq[loc-1])
      if ch >= curr:
        ch_res = (ch+1)% 20
      else:
        ch_res = ch
      for k,v in get_vocab().items():
        if v == ch_res:
          ch_res = k
          break
      if ch_res is None:
        print("WRONGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        print(loc, ch_res)
      #else:
      #  print(loc, ch, curr, seq[loc-1], ch_res)
      new_variant.append((loc, ch_res))
    return new_variant
  
  other_variants = [list(zip(np.random.randint(low=330, high=531, size=12), 
                             np.random.randint(low=19, size=12))) for i in range(1_000)]
  other_variants = list(map(lambda _ : _other_variants_with_char(seq=seq_wt, variants=_), other_variants))  
  print(other_variants[:5])
  def _replace(seq, list_of_subt):
      new_seq = ""
      for i in range(330, 531):
          flag = False
          for substitution in list_of_subt:
              if substitution[0] == i+1:
                  new_seq += substitution[1]
                  flag = True
                  break
          if not flag:
              new_seq += seq[i]
      return new_seq
  variants_seqs = list(map(lambda _: _replace(seq=seq_wt,
                                              list_of_subt=_),
                           variants))
  variants_seqs = np.array(list(map(featurize, variants_seqs)))
  
  other_variants_seqs = list(map(lambda _: _replace(seq=seq_wt,
                                                    list_of_subt=_),
                                 other_variants))
  other_variants_seqs = np.array(list(map(featurize, other_variants_seqs)))
  
  variants_escapes = list()
  other_variants_escapes = list()
  for i in range(num_of_iterations):
    cols_to_look = selections_columns
    iteration_loss = loss_weights
    if len(columns_subsets_groups) == 1:
      cols_to_look = columns_subsets_groups[0]
    #set_seed(10*i)
    X_train, y_train, X_test, y_test, cols_to_ind = get_data_w_bind(df, 
                                                                    cols_to_look,
                                                                    binding_columns, 
                                                                    full_seq=True, 
                                                                    with_transform=False,
                                                                    dropna=True,
                                                                    test_size=test_size,
                                                                    #seed=10*i
                                                                    )
    keras.backend.clear_session()

    if with_aug:
      def _indices_from_qunatiles(y, q):
        quantiles = np.quantile(y, q, axis=0)
        is_in_qunat = (y > 0.01)#quantiles)
        is_in_qunat = (y>10*quantiles)
        #print(is_in_qunat.sum(axis=0))
        return is_in_qunat.any(axis=1).squeeze().nonzero()[0]

      #ind_to_add = list()
      ind_to_add = set()
      for q, rep in qs.items():
        #if len(cols_to_look) < y_train.shape[-2]:
        #  all_reps =  _indices_from_qunatiles(y_train[:,len(cols_to_look)], q)
        all_reps =  _indices_from_qunatiles(y_train[:,
                                                    :y_train.shape[1]-len(binding_columns),
                                                    :], q)
                                                    
        #if not downsample:
        #  all_reps = np.repeat(all_reps, 
        #                       repeats=rep)
        #ind_to_add.append(all_reps)
        ind_to_add.update(set(all_reps))
        print("---", len(ind_to_add))
      aug_set = np.array(sorted(ind_to_add))# np.unique(np.concatenate(ind_to_add))
      if downsample:
        set_to_sample = sorted(set(range(len(y_train))) - set(aug_set))
        print("+++", aug_set.shape)
        print("+++", len(set_to_sample))
        downsampled = np.random.choice(a=set_to_sample,
                                       size=len(aug_set)*2,
                                       replace=True)
        inds = sorted(set(aug_set) | set(downsampled))
        X_train = X_train[inds]
        y_train = y_train[inds]
      else:
        if ros:
          size = len(set(range(len(y_train))) - set(aug_set))
          aug_set = np.random.choice(aug_set, 
                                     size=size,
                                     replace=True
                                     )
        else:
          aug_set = np.repeat(aug_set,
                              repeats=rep)
        print("zzz", aug_set.shape, X_train.shape)
        X_train = np.concatenate(
            (
            X_train,
            X_train[aug_set]
             )
        )
        y_train = np.concatenate(
            (
             y_train,
             y_train[aug_set]
             )
        )

    if binarize:
      if isinstance(binarize, bool) and binarize:
        y_train[y_train >= threshold[cols_to_look].to_numpy().reshape(-1, 1)] = 1
        y_train[y_train < threshold[cols_to_look].to_numpy().reshape(-1, 1)] = 0
        y_test[y_test >= threshold[cols_to_look].to_numpy().reshape(-1, 1)] = 1
        y_test[y_test < threshold[cols_to_look].to_numpy().reshape(-1, 1)] = 0
      else:
        y_train[y_train >= binarize] = 1
        y_train[y_train < binarize] = 0
        y_test[y_test >= binarize] = 1
        y_test[y_test < binarize] = 0
        
    if refined_bin:
      bins = np.quantile(a=np.concatenate([y_train.flatten(), y_test.flatten()]),
                         q = [i/100 for i in range(5, 100, 5)])
      y_train =  np.digitize(x=y_train, bins=bins)
      y_test =  np.digitize(x=y_test, bins=bins)
      binarize = 0.1
      if i == 0:
        print(y_train)
    if verbose == 1:
      print(f"i={i}")
    curr_hists = list()
    curr_models = list()
    sample_weights = np.ones(shape=y_train.shape)
    sample_weights = sample_weights.mean(axis=1)                                       
    for opt_num, cols in enumerate(columns_subsets_groups):
      if verbose == 1:
        print(f"cols={cols}")
      cols = [ cols_to_ind[_] for _ in cols]
      if verbose == 1:
        print(f"cols={cols}")
      binding_cols = [ cols_to_ind[_] for _ in binding_columns]
      X_train_col = X_train[:, :] 
      y_train_col = [y_train[:, _] for _ in cols+binding_cols]
      X_test_col = X_test[:, :]
      y_test_col = [y_test[:, _] for _ in cols+binding_cols]
      model = escape_and_bind_model(hidden_dims=hidden_layers,
                                    max_seq_len=201,#X_train_col.shape[-1],
                                    prob_dims=len(cols),
                                    uncertanity_layer = uncertanity_layer,
                                    regression_dim=len(binding_columns),
                                    rate=dropout)
    
    losses=None
    metrics=None
    for l in model.layers:
      l.trainable = True
    if binarize is None or not binarize:
      losses = ["binary_crossentropy" for _ in range(len(cols))]+ ["mean_squared_error" for _ in range(len(binding_columns))]
      metrics=None
      #print("Compile Params:")
      #print(f'optimizer={keras.optimizers.Adam(learning_rate=lr)}, lr={lr}, loss={["binary_crossentropy" for _ in range(len(cols))]+ ["mean_squared_error" for _ in range(len(binding_columns))]}, loss_weights={loss_weights}')
    else:
      print("BINARY")
      #import tensorflow_addons as tfa
      print("BINARY")
      #losses = [tfa.losses.SigmoidFocalCrossEntropy() for _ in range(len(cols))]+ ["mean_squared_error" for _ in range(len(binding_columns))]
      losses = ["binary_crossentropy" for _ in range(len(cols))]+ ["mean_squared_error" for _ in range(len(binding_columns))]
      metrics = [#keras.metrics.Precision(),
                               #keras.metrics.Recall(),
                 keras.metrics.AUC(name="auc-roc"),
                 keras.metrics.AUC(name="auc-pr", curve= "PR")
                 ]
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses,
                  metrics = metrics,
                  loss_weights=loss_weights)
                  
    if with_stop_call:
      if binarize and len(cols) == 1:
        callbacks = [
                     tf.keras.callbacks.EarlyStopping(monitor="val_auc-roc",
                                                      patience=3,
                                                      restore_best_weights=True)
                     ]
      else: 
        callbacks = [
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=3,
                                                      restore_best_weights=True)
                     ]
    else:
      callbacks = []
    if i ==0 :
      print(model.get_config())
    if auto_weight:
      curr_hist = list()
      val_loss_counter = 3
      curr_loss = [0 for i in range(len(cols))]
      prev_loss = [0 for i in range(len(cols))]
      if normalized and temp >= 1:
        sample_weights = softmax(sample_weights/temp)
      iteration_loss = np.ones(len(cols))/len(cols)
      curr_hist.append(model.fit(x=X_train_col, y=y_train_col,
                      batch_size=batch_size,
                      validation_data=(X_test_col, y_test_col),
                      epochs=1,
                      callbacks=callbacks,
                      #sample_weight=sample_weights,                      
                      ).history
                       )
      curr_hist.append(model.fit(x=X_train_col, y=y_train_col,
                      batch_size=batch_size,
                      validation_data=(X_test_col, y_test_col),
                      epochs=1,
                      callbacks=callbacks,
                      #sample_weight=sample_weights,
                      ).history
                       )
      #prev_weights = loss_weight
      for loss_name, val in  curr_hist[-1].items():
            if loss_name[:11] == "prob_output":
              for i in range(1,len(cols)+1):
                if f"_{i}" in loss_name:
                  curr_loss[i-1] = val[0]
                  break

      for loss_name, val in  curr_hist[-2].items():
            if loss_name[:11] == "prob_output":
              for i in range(1,len(cols)+1):
                if f"_{i}" in loss_name:
                  prev_loss[i-1] = val[0]
                  break    
      curr_loss = np.array(curr_loss)
      prev_loss = np.array(prev_loss)
      for _ in range(epochs-2):
        #if val_loss_counter == 0:
        #  break
        #print(curr_loss)
        #print(prev_loss)
        iteration_loss = curr_loss/prev_loss
        print("loss weights:", iteration_loss)
        if normalized:
          #print(iteration_loss)
          iteration_loss = softmax(iteration_loss/temp)
        print("loss weights Normalized:", iteration_loss)#, "prev losses", curr_loss, prev_loss)
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=["binary_crossentropy" for _ in range(len(cols))]+ 
                      ["mean_squared_error" for _ in range(len(binding_columns))],
                  loss_weights=iteration_loss)
        curr_hist.append(
            model.fit(x=X_train_col, y=y_train_col,
                      batch_size=batch_size,
                      validation_data=(X_test_col, y_test_col),
                      epochs=1,
                      callbacks=callbacks,
                      sample_weight=sample_weights,                      
                      ).history
            )
        prev_loss = curr_loss
        curr_loss = [0 for _ in range(len(cols))]
        for loss_name, val in  curr_hist[-1].items():
          if loss_name[:11] == "prob_output":
            for i in range(1, len(cols)+1):
              if f"_{i}" in loss_name:
                curr_loss[i-1] = val[0]
                break
            
        curr_loss = np.array(curr_loss)
        #if curr_hist[-1]['val_loss'][0] * sum(prev_weights) >= sum(loss_weights) * curr_hist[-2]['val_loss'][0]:
        #  val_loss_counter -= 1
        #else:
        #  val_loss_counter = 3

        
        #prev_weights = loss_weights
      curr_hist_temp = {
          'loss': list(),
          'val_loss': list()
          }
      for _ in curr_hist:
        curr_hist_temp['loss'].append(_['loss'][0])
        curr_hist_temp['val_loss'].append(_['val_loss'][0])
      curr_hist_temp['loss'] = np.array(curr_hist_temp['loss'])
      curr_hist_temp['val_loss']  = np.array(curr_hist_temp['val_loss'])
      curr_hist = curr_hist_temp
    else:
      #print(len(y_train_col), y_train_col[0].shape)
      curr_hist = model.fit(x=X_train_col, y=y_train_col, 
                            batch_size=batch_size,
                            validation_data=(X_test_col, y_test_col),
                            epochs=epochs,
                            callbacks=callbacks,
                            #class_weight = {1:1-np.array(y_train_col).mean(), 0:np.array(y_train_col).mean()},
                            #sample_weight=[1 if _ < 0.01 else 20 for _ in y_train_col[0].flatten()] ,
                            #sample_weight=[1-np.array(y_train_col).mean() if _ == 1 else np.array(y_train_col).mean() for _ in y_train_col[0].flatten()] ,                            
                            ).history
      print(model.layers[-1].get_weights())
      print(f"Fit Params:batch_size={batch_size}, epochs={epochs}, callbacks={callbacks}")
    if return_val_data is True:
      vals.append(([],#X_test,
       y_test, 
       model.predict(X_test)))
      trains.append(([],#X_train,
       y_train, 
       model.predict(X_train)))
    full_history[opt_num].append(curr_hist)
    curr_hists.append([curr_hist['loss'], curr_hist['val_loss']])
    curr_models.append(model)
    if return_models:
      final_models.append(curr_models)

    hists.append(curr_hists)
    variants_escapes.append(model.predict(variants_seqs))
    other_variants_escapes.append(model.predict(other_variants_seqs))
    #print("--OTHER VAR:--", other_variants_escapes)
    #print("--OTHER VAR:--", other_variants_escapes[-5:, :,-5:])
    
    keras.backend.clear_session()
  hists_temp = np.empty(shape=(num_of_iterations,
                               len(columns_subsets_groups), 
                               2,#len(hists),
                               epochs))#max([len(_) for _ in hists])))
  for i in range(len(hists)):
    for j in range(len(hists[i])):
      for k in range(len(hists[i][j])):
        for l in range(len(hists[i][j][k])):
          hists_temp[i,j,k,l] = hists[i][j][k][l]
  del(hists)

  result_dict = {
                 'history': full_history,
                 'val_data': vals,
                 'train_data': trains,
                 'variants_escapes': np.array(variants_escapes),
                 'other_variants_escapes': np.array(other_variants_escapes),
                  }
  if return_models:
    result_dict['models'] = final_models
  if save_embeddings:  
    def _featurize(orig_seq):
      vocabulary = get_vocab()
      seq = np.array([vocabulary[word] for word in orig_seq])
      return seq
    seq_wt = str(SeqIO.read('cov2_spike_wt.fasta', 'fasta').seq)
    seq_wt = seq_wt[330: 531]
    #seq_wt = _featurize(seq_wt)
    embeddings_dicts = [dict() for _ in final_models]
    embedders = [(m.input, m.layers[2].output) for m in final_models[0]]
    print(len(final_models), len(final_models[0]))
    embedders_models = [keras.Model(inputs=(_[0]), outputs=_[1]) for _ in embedders]
    for site in range(531-330):
      for mut in get_vocab():
        print(site, mut)
        if (seq_wt[site] == mut) or (mut == "Z"):
          continue
        current_wt = seq_wt[:site]+mut+seq_wt[site+1:]
        current_wt = _featurize(current_wt)
        for i, em in enumerate(embedders_models):
          print("**",i)
          embeddings_dicts[i][(site, mut)] = em.predict(current_wt)
    result_dict['embeds'] = embeddings_dicts
  result_dict['meta'] = meta
  
  
  
  return result_dict

"""## Without binding"""

column_to_select = "COV2-2082_400_escape"
correlated_column = "COV2-2094_400_escape"
negative_correlated_column = "COV2-2499_400_escape"
neutral_column = "COV2-2165_400_escape"


import pickle
from datetime import datetime
print(f"Time: {datetime.now()}")
df = pd.read_csv(r"../data/modified/starr_cocktails_and_binding.csv")
df['site'] -= 330
i = 0
lb, ub = 0, 9
for col in covid19_selections_columns[lb:ub]:
  result_column32_all_antibodies_weighted_2 =  get_experiment_data_w_bind(num_of_iterations=100,
                                                                          df=bind_and_escapes_df,#df,#bind_and_escapes_df, #df,#bind_and_escapes_df,
                                                                          batch_size=64,
                                                                          lr=1e-4,
                                                                          hidden_layers=[64],#[4, 16, 32, 16, 4],
                                                                          columns_subsets_groups=[
                                                                                                  [col],
                                                                                                 ],
                                                                          binding_columns=[],#["bind_avg"],
                                                                            #refined_bin=True,
                                                                          epochs=40, #200,#50,
                                                                          return_models=False,
                                                                          with_aug=False,
                                                                          with_sample_weight=False,
                                                                          return_val_data=True,
                                                                          qs = {0.95:19},#{0.99: 15, 0.95: 5, 0.9: 5},
                                                                          with_stop_call=True,
                                                                          auto_weight=False,#True,
                                                                          normalized=False,
                                                                          downsample = False,
                                                                          dropout=None,#0.25,
                                                                          binarize = False,#True, #False,#True,
                                                                           # binarize=0.01,
                                                                          ros=True,
                                                                          )                                                                        
  result_column32_all_antibodies_weighted_2['code'] = code
  fn = f"no_strat_{col}_cont.pkl"
  print(f" == output is {fn} ==")
  try:
    with open(fn,  "wb") as fh:
      pickle.dump({k:v for k,v in result_column32_all_antibodies_weighted_2.items() if k not in ["models"]}, fh)
    print("Success")
  except:
    print("Fail")


for col in covid19_selections_columns[lb:ub]:
  result_column32_all_antibodies_weighted_2 =  get_experiment_data_w_bind(num_of_iterations=100,
                                                                          df=bind_and_escapes_df,#df,#bind_and_escapes_df, #df,#bind_and_escapes_df,
                                                                          batch_size=64,
                                                                          lr=1e-4,
                                                                          hidden_layers=[64],#[4, 16, 32, 16, 4],
                                                                          columns_subsets_groups=[
                                                                                                  [col],
                                                                                                 ],
                                                                          binding_columns=[],#["bind_avg"],
                                                                            #refined_bin=True,
                                                                          epochs=100, #200,#50,
                                                                          return_models=False,
                                                                          with_aug=False,
                                                                          with_sample_weight=False,
                                                                          return_val_data=True,
                                                                          qs = {0.95:19},#{0.99: 15, 0.95: 5, 0.9: 5},
                                                                          with_stop_call=True,
                                                                          auto_weight=False,#True,
                                                                          normalized=False,
                                                                          downsample = False,
                                                                          dropout=None,#0.25,
                                                                          binarize = True,
                                                                          ros=True,
                                                                          )                                                                        
  result_column32_all_antibodies_weighted_2['code'] = code
  fn = f"no_strat_{col}_bin.pkl"
  print(f" == output is {fn} ==")
  try:
    with open(fn,  "wb") as fh:
      pickle.dump({k:v for k,v in result_column32_all_antibodies_weighted_2.items() if k not in ["models"]}, fh)
    print("Success")
  except:
    print("Fail")

result_column32_all_antibodies_weighted_2 =  get_experiment_data_w_bind(num_of_iterations=100,
                                                                        df=bind_and_escapes_df,#df,#bind_and_escapes_df,#df,
                                                                        batch_size=64,
                                                                        lr=1e-4,
                                                                        hidden_layers=[64],
                                                                        columns_subsets_groups=[
                                                                                                covid19_selections_columns,
                                                                                                ],
                                                                        binding_columns=[],#[],#["bind_avg"],
                                                                        epochs=100,
                                                                        return_models=True,
                                                                        with_aug=False,
                                                                        with_sample_weight=False,
                                                                        return_val_data=True,
                                                                        qs = {0.5: 1},
                                                                        with_stop_call=False,
                                                                        auto_weight=False,#True,
                                                                        normalized=False,#True,
                                                                        downsample=False,
                                                                        temp=None,#50,
                                                                        binarize=True,#True,#True,
                                                                        ros=True,
                                                                        #save_embeddings=True
                                                                        )

result_column32_all_antibodies_weighted_2['code'] = code
fn = f"multi_bin_40_dist.pkl"
print(f" == output is {fn} ==")  
#\with open("result_all_antibodies_dynamic_weight_5_percent_normalized_loss_with_bind.pkl", "wb") as fh:
try:
  with open(fn,  "wb") as fh:
    pickle.dump({k:v for k,v in result_column32_all_antibodies_weighted_2.items() if k not in ["models"]}, fh)
  print("Success")
except:
  print("Fail")
  pickle.dump({k:v for k,v in result_column32_all_antibodies_weighted_2.items() if k not in ["models"]}, fh)
  

result_column32_all_antibodies_weighted_2 =  get_experiment_data_w_bind(num_of_iterations=100,
                                                                        df=bind_and_escapes_df,#df,#bind_and_escapes_df,#df,
                                                                        batch_size=64,
                                                                        lr=1e-4,
                                                                        hidden_layers=[64],
                                                                        columns_subsets_groups=[
                                                                                                covid19_selections_columns,
                                                                                                ],
                                                                        binding_columns=[],#[],#["bind_avg"],
                                                                        epochs=100,
                                                                        return_models=True,
                                                                        with_aug=False,
                                                                        with_sample_weight=False,
                                                                        return_val_data=True,
                                                                        qs = {0.5: 1},
                                                                        with_stop_call=False,
                                                                        auto_weight=False,#True,
                                                                        normalized=False,#True,
                                                                        downsample=False,
                                                                        temp=1,
                                                                        binarize=False,#True,#True,
                                                                        ros=True,
                                                                        save_embeddings=True
                                                                        )
result_column32_all_antibodies_weighted_2['code'] = code
fn = f"multi_cont_40_dist.pkl"
print(f" == output is {fn} ==")  
models = result_column32_all_antibodies_weighted_2["models"]
try:
  with open(fn,  "wb") as fh:
    pickle.dump({k:v for k,v in result_column32_all_antibodies_weighted_2.items() if k not in ["models"]}, fh)
  print("Success")
except:
  print("Fail")
  pickle.dump({k:v for k,v in result_column32_all_antibodies_weighted_2.items() if k not in ["models"]}, fh)
  
for i in range(10):
  models[models_ind[i]][0].save(f"model_{i}")
