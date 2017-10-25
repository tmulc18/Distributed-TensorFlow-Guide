# Distributed TensorFlow Guide 
 
Currently, there are few examples of distributed TensorFlow code.  Further, the examples that do exist are often overwhelming because they don't focus on the basics of distributed computing, but rather with how to distribute an already complicated model.


This is collection of examples for help getting started with distributed computing in TensorFlow and that can act as boilerplate code.  Many of the examples focus on implementing well-known distributed training schemes, such as those available in [Distriubted Keras](https://github.com/cerndb/dist-keras) which were discussed in the author's [blog post](http://joerihermans.com/ramblings/distributed-deep-learning-part-1-an-introduction/).  The official Distributed TensorFlow guide can be found [here]( https://www.tensorflow.org/deploy/distributed). 
 
<div align="center">
<img src="imgs/data-parallelism.png" width=50%>
</div>

Almost all the examples can be run on a single machine with a CPU and all the examples only use data-parallelism (i.e. between-graph replication).

## Beginner Tutorial

See the Beginner Tutorial folder for notebooks demonstrating core concepts used in distributed TensorFlow.  The rest of the examples assume understanding of the beginner tutorial.

* `Servers.ipynb` -- basics of TensorFlow servers
* `Parameter Sever.ipynb` -- everything about parameter servers
* `Local then Global Variable.ipynb` -- creates a graph locally then make global copies of the variables Useful for graphs that do local updates before pushing global updates (e.g. DOWNPOUR, ADAG, etc.)
* `Multiple-Workers` -- contains three notebooks: one parameter server notebook and two worker notebooks  The exercise shows how global variables are communicated via the parameter server and how local updates can be made by explicitly placing ops on local devices


## Training Algorithm Examples

The complete list of examples is below. The first example, `Non-Distributed Setup`, shows the basic learning problem we want to solve distributively; this example should be familiar to all since it doesn't use any distributed code.  The second example, `Distributed Setup` shows the same problem being solved with distributed code (i.e. with one parameter server and one worker). 

* `Non-Distributed Setup`
* `Distributed Setup`
* `HogWild` (Asychronous SGD)
* `DOWNPOUR`
* `ADAG` (Asynchronous Distributed Adaptive Gradients)
* `Synchronous SGD`
* `Synchronous SGD different learning rates`
* `SDAG` (Synchronous Distributed Adaptive Gradients) **WIP**
* `Multiple GPUs Single Machine`
* `Dynamic SGD` **TODO**
* `Asynchronous Elastic Averaging SGD` (AEASGD) **TODO**
* `Asynchronous Elastic Averaging Momentum SGD` (AEAMSGD) **TODO**

## Running Training Algorithm Examples
All the training examples (except the non-distributed example) live in a folder.  To run them, move to the example directory and run the bash script.

```bash
cd <example_name>/
bash run.sh
``` 

In order to completely stop the example, you'll need to kill the python processes associated with it.  If you want to stopped training early, then there will be python processes for each of the workers in addition to the parameter server processes.  Unfortunately, the parameter server processes continue to run even after the workers are finished--these will always need to be killed manually.   To kill all python processes, run pkill.

```bash
sudo pkill python
```

## Requirements

* Python 2.7
* TensorFlow >= 1.2


## Links
* [Official Documenation](https://www.tensorflow.org/deploy/distributed)
* [Threads and Queues](https://www.tensorflow.org/programmers_guide/threading_and_queues)
* [More TensorFlow Documentation](https://www.tensorflow.org/api_guides/python/train#Distributedexecution)

## Glossary
* [Server](https://www.tensorflow.org/api_docs/python/tf/train/Server) -- encapsulates a Session target and belongs to a cluster
* [Coordinator](https://www.tensorflow.org/api_docs/python/tf/train/Coordinator) -- coordinates threads
* [Session Manager](https://www.tensorflow.org/api_docs/python/tf/train/SessionManager) -- restores session and initialized variables and coordinates threads
* [Supervisor](https://www.tensorflow.org/api_docs/python/tf/train/Supervisor) -- good for threads. Coordinater, Saver, and Session Manager. > Session Manager
* [Session Creator](https://www.tensorflow.org/api_docs/python/tf/train/SessionCreator) -- Factory for creating a session?
* [Monitored Session](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession) -- Session.  initialization, hooks, recovery.
* [Monitored Training Session](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession) -- only distributed solution for sync optimization
* [Sync Replicas](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer) -- wrapper of optimizer for synchronous optimization
* [Scaffold](https://www.tensorflow.org/api_docs/python/tf/train/Scaffold) -- holds lots of meta training settings and passed to Session creator

### Hooks
* [Stop Hook](https://www.tensorflow.org/api_docs/python/tf/train/StopAtStepHook) -- Hook to request  stop training

## Algorithm References

* [Hogwild!](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
* [DOWNPOUR](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
* [ADAG](http://joerihermans.com/ramblings/distributed-deep-learning-part-1-an-introduction/)
* [EASGD and EAMSGD](https://arxiv.org/abs/1412.6651)