from generator import GenerationEngine

engine = GenerationEngine()
print('device', engine.device, 'dtype', engine.dtype)
try:
    engine.load('sdxl-turbo')
    print('loaded', engine._loaded_key)
except Exception as exc:
    print('ERROR', type(exc).__name__, exc)
