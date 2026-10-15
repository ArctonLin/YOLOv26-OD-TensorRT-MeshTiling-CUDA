import json
import argparse
import os
import sys

def check_is_valid_fp16(onnx_path):
    try:
        import onnx
        from onnx import TensorProto
        model = onnx.load(onnx_path)
        for inp in model.graph.input:
            if inp.type.tensor_type.elem_type != TensorProto.FLOAT16:
                return False
        for out in model.graph.output:
            if out.type.tensor_type.elem_type != TensorProto.FLOAT16:
                return False
        fp16_count = sum(1 for init in model.graph.initializer if init.data_type == TensorProto.FLOAT16)
        fp32_count = sum(1 for init in model.graph.initializer if init.data_type == TensorProto.FLOAT)
        # In a genuine FP16 YOLO model, weights are FLOAT16, with at most minor float32 constants (e.g. Resize scales)
        if fp16_count == 0 or fp16_count < fp32_count:
            return False
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config.json')
    parser.add_argument('--device', type=str, default=None, help='Device to use for export (e.g. 0, cuda:0)')
    parser.add_argument('--force', action='store_true', help='Force re-export even if ONNX exists')
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

    if os.path.exists(onnx_model) and not args.force:
        if check_is_valid_fp16(onnx_model):
            print(f"ONNX model '{onnx_model}' already exists and is valid FP16. Skipping export.")
            print("Pass --force to re-export if needed.")
            return
        else:
            print(f"Existing ONNX model '{onnx_model}' does not have full FP16 weights. Re-exporting...")

    # Delay the import of torch & ultralytics
    try:
        import torch
    except ImportError:
        print("PyTorch package not found. Please activate your PyTorch environment.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics package not found. Please install it or activate venv.")
        sys.exit(1)

    # Device selection: Ultralytics requires a CUDA device for half=True ONNX export
    device = args.device or config.get('device', None)
    if device is None:
        if torch.cuda.is_available():
            device = 0
        else:
            print("Error: CUDA is not available! Ultralytics requires a CUDA device (e.g. device=0) to export ONNX with half=True (FP16 weights).")
            print("On CPU, PyTorch/Ultralytics silently falls back to FP32 weights.")
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
    print(f"Configuration: batch size={batch_size}, imgsz={(model_h, model_w)}, device={device}, half=True (FP16)")
    
    # Export to ONNX using Ultralytics with CUDA device and half=True
    exported_path = model.export(
        format='onnx',
        imgsz=(model_h, model_w),
        batch=batch_size,
        dynamic=False,
        half=True,  # Native FP16 Export (weights + IO)
        device=device
    )
    
    if exported_path and os.path.exists(exported_path):
        if os.path.abspath(exported_path) != os.path.abspath(onnx_model):
            print(f"Moving exported model from {exported_path} to {onnx_model}")
            if os.path.exists(onnx_model):
                os.remove(onnx_model)
            os.rename(exported_path, onnx_model)

        print(f"\nVerifying FP16 precision for {onnx_model}...")
        try:
            import onnx
            from onnx import TensorProto
            from collections import Counter
            model_onnx = onnx.load(onnx_model)
            
            # Print input and output information
            for inp in model_onnx.graph.input:
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                dtype_name = TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
                print(f"  Input:  '{inp.name}', shape: {shape}, dtype: {dtype_name}")
                if inp.type.tensor_type.elem_type != TensorProto.FLOAT16:
                    print(f"  WARNING: Input '{inp.name}' is {dtype_name}, expected FLOAT16!")

            for out in model_onnx.graph.output:
                shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
                dtype_name = TensorProto.DataType.Name(out.type.tensor_type.elem_type)
                print(f"  Output: '{out.name}', shape: {shape}, dtype: {dtype_name}")
                if out.type.tensor_type.elem_type != TensorProto.FLOAT16:
                    print(f"  WARNING: Output '{out.name}' is {dtype_name}, expected FLOAT16!")

            init_counter = Counter(init.data_type for init in model_onnx.graph.initializer)
            print("  Initializer dtypes:")
            for dtype_val, count in init_counter.items():
                dtype_name = TensorProto.DataType.Name(dtype_val)
                print(f"    - {dtype_name}: {count}")

            fp16_count = init_counter.get(TensorProto.FLOAT16, 0)
            fp32_count = init_counter.get(TensorProto.FLOAT, 0)
            if fp16_count > 0 and fp16_count > fp32_count:
                print(f"Verification PASSED: {fp16_count} weight initializers are natively in FLOAT16 (FP16).")
            else:
                print("WARNING: Model weights do not appear to be primarily FLOAT16!")

        except Exception as e:
            print(f"Verification warning: {e}")

        print(f"ONNX Export complete: {onnx_model}")
    else:
        print("Export failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
