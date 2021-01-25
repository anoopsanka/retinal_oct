# Setup
## 1. Check out the repo

```sh
cd retinal_oct
git pull origin main
```

## 2. Set up the Python environment


Run `conda env create -f environment.yml` to create an environment called `retina_env`.

Also, run ```export PYTHONPATH=.``` before executing any commands later on, or you will get errors like `ModuleNotFoundError: No module named ...`.

Install Python libraries using `pip-sync`, which will let us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that may not support the `conda` environment.


After creating env, activate the env and install the requirements

```sh
conda activate retina_env
pip-sync requirements.txt requirements-dev.txt
```


If you add, remove, or need to update versions of some requirements, edit the `.in` files, then run

```
pip-compile requirements.in && pip-compile requirements-dev.in
```