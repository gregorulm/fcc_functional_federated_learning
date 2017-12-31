
import sys
import time
from os import system, chdir

NUM_OF_RUNS = 5


print("Benchmarking C Node:")
chdir("./erl_cnode/")

print("Compiling")
system("gcc client_ann.c ei_utils.c fann_utils.c -o client_ann -lerl_interface -lei -lpthread -ldoublefann -lm")
system("erlc fl.erl")

for i in range(NUM_OF_RUNS):
    print("\nRun #" + str(i+1))
    system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")


chdir("../")
print("\n\nBenchmarking concurrent erlang:")
chdir("./erl/")

print("Compiling")
system("erlc fl.erl")

for i in range(NUM_OF_RUNS):
    print("\nRun #" + str(i+1))
    system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")


chdir("../")
print("\n\nBenchmarking distributed erlang:")
chdir("./erl_dist/")

print("Compiling")
system("erlc fl.erl")

for i in range(NUM_OF_RUNS):
    print("\nRun #" + str(i+1))
    system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
    system("./exit.escript 10")
    time.sleep(1)


chdir("../")
print("\n\nBenchmarking erlang with NIFs:")
chdir("./erl_nif/")

print("Compiling")
system("erlc fl.erl")

for i in range(NUM_OF_RUNS):
    print("\nRun #" + str(i+1))
    system("erl -noshell -eval 'fl:main().' -eval 'init:stop().'")


print("\nDone benchmarking.")
