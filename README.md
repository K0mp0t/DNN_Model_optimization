# Model_optimization

There are two versions (PT and TF) of the same CRNN model trained on the exact same dataset

## Results table
Character Error Rate is used as metric (so lower = better)
<pre>
+--------------------------+-------------+-----------------+------------+---------------+<br />
|  Optimization type       |    Params   | Inference time  | Loss value | Metric value  |<br />
+--------------------------+-------------+-----------------+------------+---------------+<br />
| Original model           |   1.492M    |     31.21       |  0.623181  |   0.049073    |<br />
+--------------------------+-------------+-----------------+------------+---------------+<br />
| [Channels pruning](Optimization experiments/Channels pruning.ipynb)         |   1.389M    |     25.78       |  0.904908  |   0.055323    |<br />
+--------------------------+-------------+-----------------+------------+---------------+<br />
| [Offline response-based](Optimization experiments/Response_based_distilation.ipynb)   |   0.922M    |     24.54       |  1.161216  |   0.054938    |<br />
| distilation              |             |                 |            |               |<br />
+--------------------------+-------------+-----------------+------------+---------------+<br />
</pre>
