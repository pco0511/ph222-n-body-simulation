import jax
import jax.numpy as jnp

def func(x):
    return len(x)

try:
    jaxpr = jax.make_jaxpr(func)(jnp.array(1.0))
    print("JIT 가능:", jaxpr)
except Exception as e:
    print("JIT 불가능:", e)
