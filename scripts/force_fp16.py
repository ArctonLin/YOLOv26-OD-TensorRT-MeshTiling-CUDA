import onnx
from onnx import helper, TensorProto
import sys

def force_fp16(onnx_path, out_path):
    model = onnx.load(onnx_path)
    
    # Check if we need to insert Cast nodes
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            # check the node that produces this output
            pass
            
    # The simplest way to force FP16 in strongly typed TRT is to use graphsurgeon, 
    # but let's just use polygraphy or simple onnx node insertion
    # Actually, trtexec has a flag in older versions: --fp16. In TRT 11, it's --fp16?
    # Wait! TRT 11.2 trtexec does HAVE --fp16. Earlier it said "Unknown option: --fp16 ". Notice the trailing space! 
    # In my run.sh: `trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16 `
    pass

