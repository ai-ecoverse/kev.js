// onnxruntime-node 1.30 wraps float16 tensors in Float16Array when the runtime has one (Node >= 24), but its N-API
// binding does not recognise Float16Array and reads 0 bytes. Hiding it makes ORT use Uint16Array. Import before ORT.
delete (globalThis as { Float16Array?: unknown }).Float16Array;
