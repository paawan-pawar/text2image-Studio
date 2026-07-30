from generator import GenerationEngine, GenerationRequest

engine = GenerationEngine()
print('device', engine.device, 'dtype', engine.dtype)
try:
    engine.load('sdxl-turbo')
    print('loaded', engine._loaded_key)
    req = GenerationRequest(
        prompt='A fantasy castle on a cliff',
        negative_prompt='blurry, low quality',
        model_key='sdxl-turbo',
        steps=2,
        guidance_scale=0.0,
        width=512,
        height=512,
        seed=42,
        num_images=1,
    )
    result = engine.generate(req)
    print('generated', len(result.images), 'images, seeds', result.seeds_used)
except Exception as exc:
    import traceback
    traceback.print_exc()
