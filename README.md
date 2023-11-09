# DNN model optimization

There are two versions (PT and TF) of the same CRNN model trained on the exact same dataset.
I'll try to conduct more experiments and add more notebooks to this repository (you may find my TODO list below)

## Results table
Character Error Rate is used as metric (so lower = better)
Loss function values might be incompatible between TF and PT due to possible implementation differences
All calculations performed on Goolge Colab's Tesla T4
<pre>
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
|  Optimization type       |   Runtime   |   #Params   | Inference time  | Loss value | Metric value  | Saved model size (KB)   |
|                          |             |             | (ms per batch)  |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Original model           |   TF        |   1.492M    |     53.708      |  0.293844  |   0.002215    |       17_623 (h5)       |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Original model           |   PT        |   1.492M    |     3.886       |  14.04322  |   0.049891    |  5849 (state dict .pt)  |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Channels pruning (Full)  |   PT        |   1.213M    |     3.432       |  14.10033  |   0.050777    |           ????          |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Channels pruning         |   PT        |   1.389M    |     3.161       |  14.0428   |   0.049154    |  5694 (state dict .pt)  |
| (Partial)                |             |             |                 |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Low-magnitude pruning    |   TF        |   1.213M    |     41.095      |  0.478418  |   0.009486    |           ????          |
| (Full)                   |             |             |                 |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Low-magnitude pruning    |   TF        |   1.389M    |     42.377      |  0.348688  |   0.007657    |           ????          |
| (Partial)                |             |             |                 |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| Offline response-based   |   PT        |   0.922M    |     2.026       |  14.102808 |   0.057071    |  3779 (state dict .pt)  |
| distilation              |             |             |                 |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| tf2onnx converted        | ONNXRuntime |   1.492M    |     29.454      |  0.293844  |   0.002215    |           5835          |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| torch2onnx converted     | ONNXRuntime |   1.492M    |     8.122       |  0.611737  |   0.047295    |           5849          |
| + ONNXOptimizer          |             |             |                 |            |               |                         |
| + ONNXSimplifier         |             |             |                 |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
| tf2onnx converted        | ONNXRuntime |   1.492M    |     28.541      |  0.293844  |   0.002215    |           5837          |
| + ONNXOptimizer          |             |             |                 |            |               |                         |
| + ONNXSimplifier         |             |             |                 |            |               |                         |
+--------------------------+-------------+-------------+-----------------+------------+---------------+-------------------------+
</pre>

## TODO List

* Add quantization experiments (tfmot supported, not sure about PT)
* Research non-linear inference time dependence on batch_size (TF affected, not sure about others)
* Take a look at model weights clustering (supported by tfmot, though, maybe implement my own with PT)
* Try TensorRT as onnxruntime execution provider (should give some interesting performance results)
* Recalculate all experiments with my own hardware for better results stability
* Better PT pruning algorithm (add few more steps, maybe alter channels selection algorithm)
* Enhance testing algorithms to get more stable results (maybe test TF and PT models with simgle function)
