## SSGD (Synchronous SGD)

Have workers send their updates to a ps, but only update the variables on the ps after *N* updates have been accumulated.  If the number of workers is *M* and *M>N*, then this is known as dropping the last *M-N* *stale gradients*.
