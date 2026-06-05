import torch

path = 'models/sveldman_hand_model.pt'

def main():
    print(f'inspecting {path}')
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())

    print('\ntrying torchscript')
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.jit.load(path, map_location=device)
        print('torchscript ok', type(model))
        print(model)
        return
    except Exception as exc:
        print('torchscript fail', type(exc).__name__, exc)

    print('\ntrying torch.load')
    try:
        obj = torch.load(path, map_location='cpu')
        print('torch.load ok', type(obj))
        if isinstance(obj, dict):
            print('keys', list(obj.keys())[:50])
            for key, value in list(obj.items())[:10]:
                if hasattr(value, 'shape'):
                    print(key, type(value), tuple(value.shape))
                else:
                    print(key, type(value))
        else:
            print(obj)
    except Exception as exc:
        print('torch.load fail', type(exc).__name__, exc)

if __name__ == '__main__':
    main()
