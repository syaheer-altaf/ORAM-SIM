# ORAM Simulator

The purpose of this project is to implement Path ORAM and a primitive that supports serving requests in batches. The implementation files are located at `./src`.

## To Build & Execute

Simply run the following command on terminal

```bash
make compile
```

The corresponding executables will be located at `./compiled`. The current state of the project does not support a friendlier pipeline; users are expected to change the parameters accordingly -- especially the number of accesses to ORAM -- before compilation. Then run, for example,

```bash
./compiled/oram_test
```
