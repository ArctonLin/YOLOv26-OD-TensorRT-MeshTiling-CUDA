import onnx
from onnx import helper
from onnx import TensorProto
import sys

def convert_io_to_fp16(onnx_path, out_path):
    model = onnx.load(onnx_path)
    for input in model.graph.input:
        if input.type.tensor_type.elem_type == TensorProto.FLOAT:
            input.type.tensor_type.elem_type = TensorProto.FLOAT16
    for output in model.graph.output:
        if output.type.tensor_type.elem_type == TensorProto.FLOAT:
            output.type.tensor_type.elem_type = TensorProto.FLOAT16
    
    # We might need to insert Cast nodes if the network expects FLOAT internally but we changed the input/output to FLOAT16.
    # But if the network is already exported with half=True, its internal tensors might already be FLOAT16!
    # Let's check internal nodes. If half=True did its job, the network expects FLOAT16.
    
    onnx.save(model, out_path)

if __name__ == "__main__":
    convert_io_to_fp16(sys.argv[1], sys.argv[2])
