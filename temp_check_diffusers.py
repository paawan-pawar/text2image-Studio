import diffusers
print('diffusers', diffusers.__version__)
for name in ['StableDiffusionXLPipeline','AutoPipelineForText2Image','FluxPipeline']:
    print(name, hasattr(diffusers, name))
for name in ['EulerAncestralDiscreteScheduler','EulerDiscreteScheduler','DPMSolverMultistepScheduler','DDIMScheduler','UniPCMultistepScheduler']:
    print(name, hasattr(diffusers, name))
print('available pipelines:', [n for n in dir(diffusers) if n.endswith('Pipeline')][:50])
print('available schedulers:', [n for n in dir(diffusers) if n.endswith('Scheduler')][:50])
