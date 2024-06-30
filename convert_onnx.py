import torch
import torch.nn as nn
import sys
from networks import *
from configs import parse_args_convert


def main(args):
    if args.pth_path == "":
        print("Please provide the .pth model weight path through pth_path arg.")
        sys.exit()
    if args.onnx_path == "":
        print("Please provide the .onnx model path through onnx_path arg.")
        sys.exit()
    
    model = get_model()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pth_path)['state_dict'])
    model.to('cuda')
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to('cuda')
    torch.onnx.export(model.module, (dummy_input, dummy_input), args.onnx_path, opset_version=args.opset_ver)

if __name__ == '__main__':
    args = parse_args_convert()
    main(args=args)
