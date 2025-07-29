import h5py
import json

model_path = 'best_vgg16_balanced_strategy6_1 (2).h5'

with h5py.File(model_path, 'r') as f:
    print("[INFO] Layer yang ditemukan di model:")
    for layer in f['model_weights']:
        print("-", layer)

    print("\n[INFO] Semua key di root:")
    for key in f.keys():
        print("-", key)

    if 'model_config' in f.attrs:
        raw_config = f.attrs['model_config']
        
        # FIX: decode hanya jika bytes
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode('utf-8')
        
        parsed = json.loads(raw_config)
        print("\n[INFO] Tipe model:", parsed.get('class_name'))

        print("\n[INFO] Layer Detail (nama dan tipe):")
        for i, layer in enumerate(parsed['config']['layers']):
            print(f"{i+1}. {layer['name']} ({layer['class_name']})")
            if layer['class_name'] == 'Lambda':
                print("   ⚠️  Ini Lambda layer. Detail:")
                print("   config:", layer['config'].get('name', '(tidak ada nama)'))
                print("   function:", layer['config'].get('function', '[function hidden]'))