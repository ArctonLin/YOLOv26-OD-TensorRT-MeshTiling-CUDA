import json
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config.json')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    pt_model = config.get('pt_model_path', 'yolo26n.pt')
    trt_model = config.get('model_path', 'yolo26n.engine')
    onnx_model = trt_model.replace('.engine', '.onnx')
    batch_size = config.get('batch_size', 28)
    model_w = config['model_size']['width']
    model_h = config['model_size']['height']

    if os.path.exists(onnx_model):
        print(f"ONNX model {onnx_model} already exists. Skipping export.")
        return

    # Delay the import of ultralytics so it doesn't fail if the user is just running C++ side later
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics package not found. Please install it or activate venv.")
        sys.exit(1)

    print(f"Loading {pt_model}...")
    try:
        model = YOLO(pt_model)
    except Exception as e:
        print(f"Failed to load {pt_model}: {e}")
        # fallback for testing
        print("Falling back to yolov8n.pt for testing if yolo26n.pt is missing...")
        model = YOLO('yolov8n.pt')

    print(f"Exporting to ONNX format: {onnx_model} ...")
    print(f"Configuration: batch size={batch_size}, imgsz={(model_h, model_w)}")
    
    # Export to ONNX using Ultralytics built-in method
    exported_path = model.export(
        format='onnx',
        imgsz=(model_h, model_w),
        batch=batch_size,
        dynamic=False,
        half=True  # FP16 Export
    )
    
    if exported_path and os.path.exists(exported_path):
        if os.path.abspath(exported_path) != os.path.abspath(onnx_model):
            print(f"Moving exported model from {exported_path} to {onnx_model}")
            os.rename(exported_path, onnx_model)
        print(f"Converting IO to FP16 for {onnx_model}...")
        try:
            import onnx
            from onnx import helper, TensorProto
            model_onnx = onnx.load(onnx_model)
            
            # For output, we want the graph to output FP16
            for out in model_onnx.graph.output:
                if out.type.tensor_type.elem_type == TensorProto.FLOAT or out.type.tensor_type.elem_type == TensorProto.FLOAT16:
                    orig_out_name = out.name
                    new_internal_out = orig_out_name + "_internal"
                    
                    # Find the node that produces out.name and change its output to new_internal_out
                    for node in model_onnx.graph.node:
                        for i, o in enumerate(node.output):
                            if o == orig_out_name:
                                node.output[i] = new_internal_out
                                
                    cast_node = helper.make_node("Cast", inputs=[new_internal_out], outputs=[orig_out_name], to=TensorProto.FLOAT16)
                    model_onnx.graph.node.append(cast_node)
                    out.type.tensor_type.elem_type = TensorProto.FLOAT16
                    
            # For input, YOLOv8 export already sets it to FP16 because half=True, but let's ensure the tensor info matches
            for inp in model_onnx.graph.input:
                if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                    inp.type.tensor_type.elem_type = TensorProto.FLOAT16
                    
            onnx.save(model_onnx, onnx_model)
            print("IO converted to FP16 successfully.")
        except Exception as e:
            print(f"Failed to convert IO to FP16: {e}")
        print(f"ONNX Export complete: {onnx_model}")
    else:
        print("Export failed.")

if __name__ == '__main__':
    main()
