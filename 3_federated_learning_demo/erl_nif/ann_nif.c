
#include "doublefann.h"
#include "nif_utils.h"
#include "fann_utils.h"

#define INPUT_NODES 2
#define HIDDEN_NODES 3
#define OUTPUT_NODES 2


static ERL_NIF_TERM train_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  // Handle input
  ERL_NIF_TERM weights = argv[0];
  int arity;
  const ERL_NIF_TERM *tuple;
  if(!enif_get_tuple(env, weights, &arity, &tuple) || arity != 3){
    fprintf(stderr, "Bad tuple.\n\r");
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM erl_input_weights = tuple[0];
  ERL_NIF_TERM erl_bias_weights = tuple[1];
  ERL_NIF_TERM erl_input_bias_weights;
  if(!enif_get_list_cell(env, erl_bias_weights, &erl_input_bias_weights, &erl_bias_weights)){
    fprintf(stderr, "Bad list from input bias.\n\r");
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM erl_hidden_bias_weights;
  if(!enif_get_list_cell(env, erl_bias_weights, &erl_hidden_bias_weights, &erl_bias_weights)){
    fprintf(stderr, "Bad list from hidden bias.\n\r");
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM erl_hidden_weights = tuple[2];

  ERL_NIF_TERM training_data = argv[1];

  struct fann *ann = fann_create_standard(3, INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);

  // decode weights
  double** input_weights = list2D_to_array2D(env, erl_input_weights);
  double** hidden_weights = list2D_to_array2D(env, erl_hidden_weights);

  double* input_bias_weights = list_to_array(env, erl_input_bias_weights);
  double* hidden_bias_weights = list_to_array(env, erl_hidden_bias_weights);

  // update weights to ann
  update_weights(ann, input_weights, hidden_weights
    , HIDDEN_NODES, INPUT_NODES
    , OUTPUT_NODES, HIDDEN_NODES
    , input_bias_weights, hidden_bias_weights);

  // decode training data
  unsigned int input_len;
  if(!enif_get_list_length(env, training_data, &input_len)) {
    return enif_make_badarg(env);
  }
  fann_type *input = malloc(sizeof(fann_type)*input_len*INPUT_NODES);
  fann_type *output = malloc(sizeof(fann_type)*input_len*OUTPUT_NODES); //(fann_type*)calloc(input_len*INPUT_NODES, sizeof(fann_type));
  if(!train_data_to_arrays(env, training_data, input, output)){
    fprintf(stderr, "Failed to convert training data.\n\r" );
  }

  // train
  struct fann_train_data *data = fann_create_train_array(input_len,
      INPUT_NODES, input, OUTPUT_NODES, output);

  fann_set_callback(ann, &test_callback);
  fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);

  const unsigned int n_epochs = 1;
  const unsigned int report_interval = 0;
  const float learning_rate = 0.01;
  fann_train_on_data(ann, data, n_epochs, report_interval, learning_rate);

  // extract weights from ann
  extract_weights(ann, input_weights, hidden_weights
            , HIDDEN_NODES, INPUT_NODES
            , OUTPUT_NODES, HIDDEN_NODES
            , input_bias_weights, hidden_bias_weights);

  // encode trained weights
  erl_input_weights = array2D_to_list2D(env, input_weights, HIDDEN_NODES, INPUT_NODES);
  erl_hidden_weights = array2D_to_list2D(env, hidden_weights, OUTPUT_NODES, HIDDEN_NODES);
  erl_input_bias_weights = array_to_list(env, input_bias_weights, HIDDEN_NODES);
  erl_hidden_bias_weights = array_to_list(env, hidden_bias_weights, OUTPUT_NODES);

  ERL_NIF_TERM biases = enif_make_list2(env, erl_input_bias_weights, erl_hidden_bias_weights);
  ERL_NIF_TERM result = enif_make_tuple3(env, erl_input_weights, biases, erl_hidden_weights);

  // free everything
  fann_destroy(ann);
  fann_destroy_train(data);
  for(int i = 0; i < HIDDEN_NODES; i++){
    free(input_weights[i]);
  }
  for(int i = 0; i < OUTPUT_NODES; i++){
    free(hidden_weights[i]);
  }
  free(input_weights); free(hidden_weights);
  free(input_bias_weights); free(hidden_bias_weights);
  free(input); free(output);

  return result;
}



static ErlNifFunc nif_funcs[] = {
  {"train", 2, train_nif},
};

ERL_NIF_INIT(fl, nif_funcs, NULL, NULL, NULL, NULL)
