First load the modules for python (>=3.12), gcc and cuda (>=12.6.0)
python -m venv sparc
source sparc/bin/activate

```
pip install vllm
pip install trl
pip install trl[vllm]
pip install sparc-puzzle
pip install psutil
pip install flash_attn --no-build-isolation
```

Symlink the models folder to your scratch disk with more disk memory
```
ln -s /scratch/..../models $HOME/sparc-train/models
``