# Model_optimization

There are two versions (PT and TF) of the same CRNN model trained on the exact same dataset

## Results table
Character Error Rate is used as metric (so lower = better)
<pre>
+--------------------------+-------------+-----------------+------------+---------------+
|  Optimization type       |    Params   | Inference time  | Loss value | Metric value  |
+--------------------------+-------------+-----------------+------------+---------------+
| Original model           |   1.492M    |     31.21       |  0.623181  |   0.049073    |
+--------------------------+-------------+-----------------+------------+---------------+
| Channels pruning         |   1.389M    |     25.78       |  0.904908  |   0.055323    |
+--------------------------+-------------+-----------------+------------+---------------+
| Offline response-based   |   0.922M    |     24.54       |  1.161216  |   0.054938    |
| distilation              |             |                 |            |               |
+--------------------------+-------------+-----------------+------------+---------------+
| ONNX conversion          |   1.492M    |     12.30       |  0.611737  |   0.047295    |
| + ONNXOptimizer          |             |                 |            |               |
| + ONNXSimplifier         |             |                 |            |               |
+--------------------------+-------------+-----------------+------------+---------------+
</pre>
