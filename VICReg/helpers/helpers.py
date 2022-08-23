#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dotmap import DotMap
from json import load as json_load_from_file


# In[ ]:


def load_config(path="config.json"):  # pragma: no cover
    with open(path) as f:
        config_dict = json_load_from_file(f)
    return DotMap(config_dict)

