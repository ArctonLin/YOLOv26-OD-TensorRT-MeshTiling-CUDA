import sys
import onnx
from onnxconverter_common import float16

onnx_model = onnx.load(sys.argv[1])
onnx_model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
onnx.save(onnx_model_fp16, sys.argv[2])
