Functional Federated Learning
=============================
2017 Fraunhofer-Chalmers Centre for Industrial Mathematics

Demo and benchmarking code supporting the paper by G. Ulm, E. Gustavsson,
M. Jirstrand


Implementation:
---------------
Gregor Ulm     (gregor.ulm@fcc.chalmers.se)
Adrian Nilsson (adrian.nilsson@fcc.chalmers.se)
Simon Smith    (simon.smith@fcc.chalmers.se)


Content:
--------
/foundation    inital Erlang code
/erl           toy system completely based on Erlang
/erl_dist      toy system completely based on distributed Erlang
/erl_cnode     toy system using C nodes for computations
/erl_nif       toy system using C codes as NIFs for computations
/other         scripts for visualiation and benchmarking


Licence:
--------
Code contribution by FCC are released under the MIT license. Part of the
implementation uses the C library FANN, which is under the GNU Lesser
General Public License v2.1.
