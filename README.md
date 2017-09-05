# Distributed TensorFlow Examples
Currently, there are few examples of distributed TensorFlow code.  Further, the examples that do exist are often overwhelming because they don't focus on the basics of distributed computing, but rather with how to distribute an already complicated model.

This is collection of examples for help getting started with distributed computing in TensorFlow and that can act as boilerplate code.  Many of the examples focus on implementing well-known distributed training schemes, such as those available in [Distriubted Keras](https://github.com/cerndb/dist-keras).  The official Distributed TensorFlow guide can be found [here]( https://www.tensorflow.org/deploy/distributed).  Almost all the examples can be run on a single machine with a CPU. 


# Examples

The complete list of examples is below.  The asynchronous examples are *easier* than the synchronous, so people getting started should first have a complete understanding of those before moving to synchronous examples.  The first example, `Non-Distributed Setup`, shows the basic learning problem we want to solve distributively; this example should be familiar to all since it doesn't use any distributed code.  The second example, `Distributed Setup` shows the same problem being solved with distributed code (i.e. with one parameter server and one worker). 

* `Non-Distributed Setup`
* `Distributed Setup`
* `HogWild` (Asychronous SGD)
* `Synchronous SGD`
* `Synchronous SGD variables learning rates`
* `ADAG` (Aggregated Distributed Asynchronous Gradients) (TODO)
* `ADSG` (Aggregated Distributed Synchronous Gradients)
* `Multiple-machine single-GPU`
* `Single-machine multiple-GPU`
* `Multiple-machine multiple-GPU`
* **Dynamic SGD (TODO)**
* **Asynchronous Elastic Averaging SGD (AEASGD) (TODO)**
* **Asynchronous Elastic Averaging Momentum SGD (AEAMSGD) (TODO)**

# Running Examples
All the examples (except the non-distributed example) live in a folder.  To run them, move to the example directory and run the bash script.

```bash
cd <example_name>/
bash run.sh
``` 

In order to completely stop the example, you'll need to kill the python processes associated with it.  If you want to stopped training early, then there will be python processes for each of the workers in addition to the parameter server processes.  Unfortunately, the parameter server processes continue to run even after the workers are finished--these will always need to be killed manually.   To kill all python processes, run pkill.

```bash
sudo pkill python
```

# Requirements

* Python 2.7
* TensorFlow >= 1.2

# License

