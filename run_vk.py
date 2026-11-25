# Same as main.py but forces the Vulkan backend.
# Metal needs full Xcode for its shader compiler, Vulkan doesn't.
# usage: uv run python run_vk.py 4_3 scenes/hw4/cloth.py
import sys, runpy, slangpy as spy

_orig = spy.create_device
def create_device(*a, **kw):
    kw.setdefault('type', spy.DeviceType.vulkan)
    return _orig(*a, **kw)
spy.create_device = create_device

sys.argv = ['main.py'] + sys.argv[1:]
runpy.run_path('main.py', run_name='__main__')
