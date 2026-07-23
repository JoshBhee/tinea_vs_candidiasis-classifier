import h5py
import json

MODEL_PATH = "skin_model.h5"

with h5py.File(MODEL_PATH, 'r+') as f:
    model_config = f.attrs.get('model_config')
    if model_config is None:
        print("No model_config found in this file.")
    else:
        config_str = model_config if isinstance(model_config, str) else model_config.decode('utf-8')
        config_dict = json.loads(config_str)

        def fix_layer(layer):
            if layer.get('class_name') == 'InputLayer':
                cfg = layer['config']
                if 'batch_shape' in cfg:
                    cfg['batch_input_shape'] = cfg.pop('batch_shape')
                cfg.pop('optional', None)

        for layer in config_dict['config']['layers']:
            fix_layer(layer)

        new_config_str = json.dumps(config_dict)
        f.attrs.modify('model_config', new_config_str)

print("Done. Model config patched successfully.")