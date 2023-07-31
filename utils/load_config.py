import yaml
def config_loader(config_file):
    config = {}
    with open(config_file, "r") as stream:
        try:
             config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
             print(exc)
    return config

# conf = config_loader("configs\config.yaml")
# print(conf)


