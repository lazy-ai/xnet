# XNET

An Enhanced feedforward deep nerual network forward/inference library. 
Protobuf is used for model io, since protobuf is good at flexibility and model compatibility.
The system has the same function with [Net](https://github.com/robin1001/net), Support both float and 8bits quantized model. 
Refer [https://github.com/robin1001/net](https://github.com/robin1001/net) for performance details.

## Model Size

One main concern is the model size, since we want smaller model size, epecially for embeding device.
Here is the comparison of Net/Xnet model size.

| Model | Net(K)  | XNet(K) |
|-------|---------|---------|
| Float | 2616.12 | 2616.25 |
| 8bits | 657.14  | 1072.65 |

we can see for float model, the model size is close. 
However, for 8bits model, XNet is much larger than Net. 
Because protobuf use varint encoding for integers(8/16/32), it's useful for compression, especially for int32, but not for int8.





