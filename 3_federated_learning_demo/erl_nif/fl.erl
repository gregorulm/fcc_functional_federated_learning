-module(fl).
-compile([export_all]).
-on_load(init/0).

init() ->
    ok = erlang:load_nif("./ann_nif", 0).

train(_ModelW, _Data) ->
    exit(nif_library_not_loaded).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Federated Learning: Artifician Neural Network
% (c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
%          Mathematics, Department of Systems and Data Analysis
%
% Research and development:
% Gregor Ulm      - gregor.ulm@fcc.chalmers.se
% Emil Gustavsson - emil.gustavsson@fcc.chalmers.se
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Federated Learning (cf. McMahan et al., 2017) is a decentralized, i.e.
% distributed, approach to Machine Learning. This implementation of the
% general idea behind Federated Learning is one of if not the first
% publicly available one, and also the first public implementation in
% a functional programming language (Erlang).
%
% Federated Learning consists of the following steps:
% - select a subset of clients
% - send the current model to each client
% - for each client, update the provided model based on local data
% - for each client, send updated model to server
% - aggregate the client models, for instance by averaging, in order to
%   construct an improved global model
%
% This demo illustrates Federated Learning with 25 clients, where each
% clent uses an artificial neural network.
%
% Execution:
% Launch the Erlang/OTP 18 shell with 'erl', compile the source with
% 'c(demo).', and execute it with 'ann:main()'.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Client Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% objective function
sigmoid(X) -> 1.0 / (1.0 + math:exp(-X)).



% multiply list of inputs with list of lists of weights
forward(_    , []      , Acc) -> lists:reverse(Acc);
forward(Input, [W | Ws], Acc) ->

  Val = lists:sum(
          lists:zipwith(fun(X, Y) -> X * Y end, Input, W)),

  forward(Input, Ws, [Val | Acc]).



% compute error in output layer
output_error(Vals, Targets) ->

  lists:zipwith(
    fun(X, Y) -> X * (1.0 - X) * (Y - X) end, Vals, Targets).



client(N) ->

  receive
  % receive current model from server
  { assignment, Model_Ws, Server_Pid, Data } ->

    New_Weights = train(Model_Ws, Data),
    Server_Pid ! { update, self(), New_Weights },
    client(N)

  end.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Server process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% pick a random subset of clients;
% every client Pid has an even chance of being selected
% input is a list of tuples: [{Pid, Number}]
random_subset_guard(Pids) ->

  Val = random_subset(Pids, []),

  case length(Val) of
    0 -> random_subset_guard(Pids);
    _ -> Val
  end.



random_subset([]     , Acc) -> Acc;
random_subset([H | T], Acc) ->

  case rand:uniform(2) of
    1 -> random_subset(T, Acc);
    2 -> random_subset(T, [H | Acc])
  end.



sum_up([]        , []        , Acc) -> lists:reverse(Acc);
sum_up([Xs | Xss], [Ys | Yss], Acc) ->

  A = lists:zipwith(fun(X, Y) -> X + Y end, Xs, Ys),
  sum_up(Xss, Yss, [A | Acc]).



fold_lists([]        , Acc) -> Acc;
fold_lists([Xs | Xss], Acc) ->
  Acc_ = sum_up(Xs, Acc, []),
  fold_lists(Xss, Acc_).



scale_nested_list(XXs, N) -> [ [ X * N || X <- Xs ] || Xs <- XXs ].



server(Client_Pids_Nums, Data, Data_List, Model_Ws, N, StartTime, Stats) ->

  % send model to arbitrary subset of clients
  Subset = random_subset_guard(Client_Pids_Nums),

  % send assignment
  lists:map(
    fun({X, Num}) ->
      X ! { assignment, Model_Ws, self(), maps:get(Num, Data)} end,
      Subset),

  Subset_pids = lists:map(fun({X, _}) -> X end, Subset),

  % receive locally updated models from clients
  Vals = [
    receive { update, Pid, Val } -> Val end || Pid <- Subset_pids ],

  % separate received data
  {W_In, W_B, W_Out} = lists:unzip3(Vals),

  % Accumulators (second argument of fold_lists) starts with head of
  Sum_Local_Ws = lists:map(
    fun(L) -> fold_lists(tl(L), hd(L)) end, [W_In, W_B, W_Out] ),

  % aggregate and average:

  % existing weight: curent model
  Total        = length(Client_Pids_Nums),
  Num_Active   = length(Subset),
  Num_Inactive = Total - Num_Active,

  % scale up: Weights * Num_Inactive
  Scaled_Ws = lists:map(
      fun(X) -> scale_nested_list(X, Num_Inactive) end,
      tuple_to_list(Model_Ws)),

  Total_Ws_Sum = lists:map(
    fun({X, Y}) -> fold_lists([X], Y) end,
    lists:zip(Sum_Local_Ws, Scaled_Ws)),

  % scale down: divide by length(Client_Pids)
  Factor = 1.0 / length(Client_Pids_Nums),

  Model_Ws_ = list_to_tuple(
   lists:map(fun(X) -> scale_nested_list(X, Factor) end, Total_Ws_Sum)),

  % compute errors
  Output_Error = error_forward(Model_Ws_, Data_List, []),

  Eps = 0.0015, % pick a smaller value to extend training duration

  case all_good(Eps, Output_Error) of
    % training completed:
    true -> io:format("Done!~n", []),
            Stats;

    false ->

      case N rem 500 == 0 of

        false -> server(Client_Pids_Nums, Data, Data_List,
                        Model_Ws_, N+1, StartTime, Stats);

        true -> io:format("N: ~p~n~n", [N]),

                {MegaSec, Sec, Micro} = os:timestamp(),
                Time = (MegaSec*1000000 + Sec)*1000 + round(Micro/1000),

                Val = {Time},
                Stats_ = maps:put(N, Val, Stats),

                case os:system_time(seconds) - StartTime >= 10*60 of
                  true  -> Stats_;
                  false -> server(Client_Pids_Nums, Data, Data_List,
                                  Model_Ws_, N+1, StartTime, Stats_)
                end
      end

    end.



all_good(_  , []        ) -> true;
all_good(Eps, [Xs | Xss]) ->
  case lists:all(fun(G) -> abs(G) < Eps end, Xs) of
    true  -> all_good(Eps, Xss);
    false -> false
  end.



error_forward(_    , []    , Acc) -> Acc;
error_forward(Model, [D|Ds], Acc) ->

  {Input, Target}             = D,
  %{W_Input, W_Bias, W_Hidden} = Model,
  {W_Input, [W_B_In, W_B_Out], W_Hidden} = Model,

  Hidden_In  = forward(Input, W_Input, []),

  % add bias
  Hidden_In_ =
    lists:zipwith(fun(X, Y) -> X + Y end, Hidden_In, W_B_In),

  Hidden_Out = lists:map(fun(X) -> sigmoid(X) end, Hidden_In_),

  Output_In  = forward(Hidden_Out, W_Hidden, []),
  Output_In_ = lists:zipwith(fun(X, Y) -> X + Y end, Output_In, W_B_Out),
  Output_Out = lists:map(fun(X) -> sigmoid(X) end, Output_In_),

  A = output_error(Output_Out, Target),
  error_forward(Model, Ds, [A|Acc]).





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entry point
% run "demo:main()" to start demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create test data
generate_Data(0, Acc1, Acc2) -> {Acc1, Acc2};
generate_Data(N, Acc1, Acc2) ->
  % generate 250 entries for each client
  Vals = generate_helper(250, []),
  generate_Data(N-1, maps:put(N, Vals, Acc1), Acc2 ++ Vals).



generate_helper(0, Acc) -> Acc;
generate_helper(N, Acc) ->
  X      = random:uniform(),
  Y      = random:uniform(),
  Input  = [X, Y],
  Target_1 = math:sqrt(X * Y),
  Target_2 = math:sqrt(Target_1),
  Target = [Target_1, Target_2],
  generate_helper(N-1, [{Input, Target}|Acc]).



spawn_clients(0, Acc) -> Acc;
spawn_clients(N, Acc) ->
  Pid = spawn(?MODULE, client, [N]),
  % N: client number, for data retrieval
  spawn_clients(N-1, [{Pid, N} | Acc]).



main() ->

  io:format("Federated Learning Demo~n~n", []),

  io:format("Generating training data..~n~n", []),
  Num_Clients = 10,
  {Data, Data_List} = generate_Data(Num_Clients, maps:new(), []),
  % clients retrieve data according to dict key
  % data is sent together with model, i.e. for client N, send
  % {Model, maps:get(N, Data)} from server

  io:format("Spawning clients...~n~n", []),
  Client_Pids = spawn_clients(Num_Clients, []),

  % hardcoded input: 2 input, 3 hidden, 2 output neurons
  % 1 bias node for hidden layer

  W_Input  = [[0.2, 0.1], [0.4, 0.8], [0.7, 0.6]],
  W_Bias   = [[0.1, 0.1, 0.1], [0.1, 0.1]],
  W_Hidden = [[0.3, 0.4, 0.2], [0.05, 0.20, 0.7]],
  % alternatively, randomize weights to a small number

  % train until target error rate reached
  Model_Ws = {W_Input, W_Bias, W_Hidden},

  N     = 0, % iteration count
  Stats = maps:new(), % key: Iteration, value: {time, total abs. error}

  StartTime = os:system_time(seconds),
  Res   = server(
            Client_Pids, Data, Data_List, Model_Ws, N, StartTime, Stats),

  % process and write results to file
  Content = extract_content(lists:sort(maps:keys(Res)), Res, []),
  write_results("data/output_erl_nif"
                ++ integer_to_list(StartTime) ++ ".csv", Content).



write_results(File, Xs) ->
  {ok, S} = file:open(File, write),
  lists:foreach(
    fun({X, Y, Z}) -> io:format(S, "~p,~p,~p~n",[X, Y, Z]) end, Xs),
  file:close(S).



extract_content([]    , _  , Acc) -> lists:reverse(Acc);
extract_content([K|Ks], Map, Acc) ->
  {Time, Error} = maps:get(K, Map),
  extract_content(Ks, Map, [{K, Time, Error}|Acc]).
