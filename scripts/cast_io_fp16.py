import onnx
from onnx import helper, TensorProto
import sys

def cast_io_fp16(onnx_path, out_path):
    model = onnx.load(onnx_path)
    
    # Create new inputs (FP16)
    new_inputs = []
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT or inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
            new_inp_name = inp.name + "_fp16"
            new_inp = helper.make_tensor_value_info(new_inp_name, TensorProto.FLOAT16, [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim])
            new_inputs.append(new_inp)
            
            cast_node = helper.make_node("Cast", inputs=[new_inp_name], outputs=[inp.name], to=TensorProto.FLOAT16) # YOLO internal uses FP16 because half=True
            model.graph.node.insert(0, cast_node)
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
            # Actually, instead of renaming, let's just insert Cast from the original name to a temp name, and rename the original input in the graph
        else:
            new_inputs.append(inp)

    # For output, we want the graph to output FP16
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT or out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            orig_out_name = out.name
            new_internal_out = orig_out_name + "_internal"
            
            # Find the node that produces out.name and change its output to new_internal_out
            for node in model.graph.node:
                for i, o in enumerate(node.output):
                    if o == orig_out_name:
                        node.output[i] = new_internal_out
                        
            cast_node = helper.make_node("Cast", inputs=[new_internal_out], outputs=[orig_out_name], to=TensorProto.FLOAT16)
            model.graph.node.append(cast_node)
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
            
    onnx.save(model, out_path)

if __name__ == "__main__":
    cast_io_fp16(sys.argv[1], sys.argv[2])
