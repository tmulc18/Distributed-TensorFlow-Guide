## AGN (Accumulated Gradient Normalization) 

This method was formerly known as ADAG (Asynchronous Distributed Adaptive Gradients).

Similar to DOWNPOUR expect that it uses a communications window *T* and accumulates gradients for *T* steps before sending updates to the parameter server.
