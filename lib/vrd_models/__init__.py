from .DSRModel import DSRModel

def VRDModel(model_args):
  args = model_args
  model_constructor_str = model_args.pop("type")

  return eval(model_constructor_str)(args)

# # https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
