# ORAM Simulator

The purpose of this project is to analyze a variant of Path ORAM with non-uniform bucket sizes that are level-dependent through some experiments. The ORAM variant can also serve requests in batches. Users are expected to run it on Unix-like (e.g., Linux, macOS) operating systems.

## To Build & Execute

Simply run the following command on terminal

```bash
make compile
```

The corresponding executables will be located at `./compiled`. These executables are experiments to run and subsequently used for analyses. The current state of the project does not support a friendlier pipeline; users are expected to change the parameters accordingly -- especially the number of accesses to ORAM -- before compilation. Then run, for example,

```bash
./compiled/experiment_1
```

The results would be logged into a `.csv` file at `./results/exp1`.

## Analysis

There are several python scripts used in the analysis of the corresponding experiments. For example, one may run

```bash
python3 ./python_scripts/analyze_exp1.py
```

Any figures generated will be in that directory.